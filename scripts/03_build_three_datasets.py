
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"

RUN_IDS = [
    "20261001",
    "20261012",
    "20261015",
    "20261016",
    "20261020",
    "20261022",
    "20261029",
    "20261111",
]

STORAGE_RESOURCE_ID = 17
STORAGE_BUF_NO = 1
PICK_SORT_BUF_NO = 3
BOX_TYPE_PNO = 29

EXTRACTION_STEP_NO = 10

# tblstepparameter
PARAM_SOURCE_BOX = 3
PARAM_SOURCE_POS = 4
PARAM_TARGET_BOX = 6
PARAM_TARGET_POS = 7
PARAM_MAIN_ORDER = 8
PARAM_REQUIRED_PNO = 11

MIN_BOX_ID = 1
MAX_BOX_ID = 40
BOX_POSITIONS = range(1, 9)

PART_IDS = [800, 811, 812, 813, 814, 831, 832, 833, 834]
BOM_PARTS = [800, 814, 813, 811, 831]

INVALID_MAIN_ORDERS = {7962}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# BASIC HELPERS
# ============================================================

def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default

    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def as_datetime(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    value = str(value).strip().replace("Z", "+00:00")

    if value.endswith("+0000"):
        value = value[:-5] + "+00:00"

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def get_timestamp(row: dict[str, Any]) -> datetime | None:
    return as_datetime(row.get("timestamp"))


def load_table(run_id: str, table: str) -> pl.DataFrame:
    folder = RAW_DIR / run_id / table
    files = sorted(folder.rglob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {folder}"
        )

    frames = []

    for file in files:
        frame = pl.read_parquet(file)

        if "run_id" not in frame.columns:
            frame = frame.with_columns(
                pl.lit(run_id).alias("run_id")
            )

        if "source_file" not in frame.columns:
            frame = frame.with_columns(
                pl.lit(file.name).alias("source_file")
            )

        frames.append(frame)

    return pl.concat(frames, how="diagonal_relaxed")


def validate_columns(
    run_id: str,
    table_name: str,
    df: pl.DataFrame,
    required_columns: set[str],
) -> None:
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{run_id}/{table_name} missing columns: "
            f"{sorted(missing)}"
        )


# ============================================================
# BOX COMPOSITION FEATURES
# ============================================================

def add_box_composition_features(
    df: pl.DataFrame,
) -> pl.DataFrame:

    pno_columns = [
        f"pos_{pos}_PNo"
        for pos in BOX_POSITIONS
    ]

    df = df.with_columns(
        pl.sum_horizontal([
            (pl.col(col) != 0).cast(pl.Int64)
            for col in pno_columns
        ]).alias("occupied_positions")
    )

    df = df.with_columns(
        (8 - pl.col("occupied_positions")).alias(
            "empty_positions"
        ),
        (pl.col("occupied_positions") / 8).alias(
            "occupancy_ratio"
        ),
    )

    df = df.with_columns(
        pl.concat_list([
            pl.col(col)
            for col in pno_columns
        ])
        .list.eval(
            pl.element().filter(pl.element() != 0)
        )
        .list.unique()
        .list.len()
        .alias("unique_part_count")
    )

    for part in PART_IDS:
        df = df.with_columns(
            pl.sum_horizontal([
                (pl.col(col) == part).cast(pl.Int64)
                for col in pno_columns
            ]).alias(f"count_PNo_{part}")
        )

    df = df.with_columns(
        pl.sum_horizontal([
            pl.col(f"count_PNo_{part}")
            for part in BOM_PARTS
        ]).alias("relevant_piece_count")
    )

    df = df.with_columns(
        (
            pl.col("occupied_positions")
            - pl.col("relevant_piece_count")
        ).alias("irrelevant_piece_count")
    )

    df = df.with_columns(
        pl.sum_horizontal([
            (pl.col(f"count_PNo_{part}") > 0).cast(pl.Int64)
            for part in BOM_PARTS
        ]).alias("relevant_types_covered")
    )

    return df


# ============================================================
# BOX FEATURES AT DECISION TIME
# ============================================================

def build_box_features(
    box_id: int,
    box_state: dict[tuple[int, int], dict[str, Any]],
    order_hierarchy: dict[int, int],
) -> dict[str, Any]:

    result: dict[str, Any] = {
        "BoxID": box_id
    }

    parts = set()
    orders = set()
    main_orders = set()

    for pos in BOX_POSITIONS:
        record = box_state.get((box_id, pos), {})

        pno = as_int(record.get("PNo"))
        ono = as_int(record.get("ONo"))
        opos = as_int(record.get("OPos"))

        result[f"pos_{pos}_PNo"] = pno
        result[f"pos_{pos}_ONo"] = ono
        result[f"pos_{pos}_OPos"] = opos

        if pno != 0:
            parts.add(pno)

        if ono > 0:
            orders.add(ono)

            if ono in order_hierarchy:
                main_orders.add(order_hierarchy[ono])

    result["part_numbers"] = ";".join(
        map(str, sorted(parts))
    )

    result["order_numbers"] = ";".join(
        map(str, sorted(orders))
    )

    result["main_order_numbers"] = ";".join(
        map(str, sorted(main_orders))
    )

    return result


# ============================================================
# BUILD EXTRACTION DECISIONS
# ============================================================

def build_extraction_operations(
    step_df: pl.DataFrame,
    run_id: str,
) -> tuple[list[dict[str, Any]], int]:

    """
    One decision per (ONo, OPos, StepNo=10).

    ParameterNo 3 = selected source BoxID.
    ParameterNo 4 = selected source BoxPos.

    The decision timestamp is the first positive source-selection
    write. Storage is reconstructed strictly before this timestamp.

    ParameterNo 11 = demanded PNo.
    ParameterNo 8 = optional MainONo metadata.
    """

    required_columns = {
        "timestamp",
        "action",
        "WPNo",
        "StepNo",
        "ONo",
        "OPos",
        "ParameterNo",
        "Parameter",
    }

    validate_columns(
        run_id,
        "tblstepparameter",
        step_df,
        required_columns,
    )

    operations: dict[
        tuple[int, int, int],
        list[tuple[int, dict[str, Any]]]
    ] = {}

    for row_index, row in enumerate(step_df.to_dicts()):

        action = str(row.get("action", "")).lower()

        if action not in {"insert", "update"}:
            continue

        step_no = as_int(row.get("StepNo"), -1)

        if step_no != EXTRACTION_STEP_NO:
            continue

        parameter_no = as_int(
            row.get("ParameterNo"),
            -1,
        )

        if parameter_no not in {
            PARAM_SOURCE_BOX,
            PARAM_SOURCE_POS,
            PARAM_MAIN_ORDER,
            PARAM_REQUIRED_PNO,
        }:
            continue

        timestamp = get_timestamp(row)

        if timestamp is None:
            continue

        ono = as_int(row.get("ONo"), -1)
        opos = as_int(row.get("OPos"), -1)

        if ono <= 0 or opos < 0:
            continue

        # Match the SQL update identity:
        # ONo, OPos, StepNo.
        key = (
            ono,
            opos,
            step_no,
        )

        operations.setdefault(key, []).append(
            (row_index, row)
        )

    decisions = []
    rejected_operations = 0

    for key, indexed_rows in operations.items():

        indexed_rows.sort(
            key=lambda item: (
                get_timestamp(item[1]),
                item[0],
            )
        )

        source_box_row = None
        source_pos_row = None

        # Find the first positive source-selection writes.
        for _, row in indexed_rows:

            parameter_no = as_int(
                row.get("ParameterNo"),
                -1,
            )

            value = as_int(row.get("Parameter"))

            if value <= 0:
                continue

            if (
                parameter_no == PARAM_SOURCE_BOX
                and source_box_row is None
            ):
                source_box_row = row

            elif (
                parameter_no == PARAM_SOURCE_POS
                and source_pos_row is None
            ):
                source_pos_row = row

        if source_box_row is None or source_pos_row is None:
            continue

        source_box = as_int(
            source_box_row.get("Parameter")
        )

        source_pos = as_int(
            source_pos_row.get("Parameter")
        )

        box_timestamp = get_timestamp(source_box_row)
        pos_timestamp = get_timestamp(source_pos_row)

        if box_timestamp is None or pos_timestamp is None:
            rejected_operations += 1
            continue

        # The first write marks the beginning of the selection
        # registration. The snapshot must precede it.
        decision_timestamp = min(
            box_timestamp,
            pos_timestamp,
        )

        if not MIN_BOX_ID <= source_box <= MAX_BOX_ID:
            rejected_operations += 1
            continue

        if source_pos not in BOX_POSITIONS:
            rejected_operations += 1
            continue

        # Retrieve demand and MainONo only from values available
        # at or before the decision timestamp.
        values_before_decision: dict[int, int] = {}

        for _, row in indexed_rows:

            timestamp = get_timestamp(row)

            if timestamp is None or timestamp > decision_timestamp:
                continue

            parameter_no = as_int(
                row.get("ParameterNo"),
                -1,
            )

            if parameter_no not in {
                PARAM_MAIN_ORDER,
                PARAM_REQUIRED_PNO,
            }:
                continue

            values_before_decision[parameter_no] = as_int(
                row.get("Parameter")
            )

        required_pno = values_before_decision.get(
            PARAM_REQUIRED_PNO,
            0,
        )

        main_order = values_before_decision.get(
            PARAM_MAIN_ORDER,
            0,
        )

        # Demand is required to reconstruct feasible candidates.
        if required_pno not in BOM_PARTS:
            rejected_operations += 1
            continue

        # Keep the existing exclusion only when MainONo is known.
        if main_order in INVALID_MAIN_ORDERS:
            continue

        # WPNo is metadata, not part of the operation identity.
        wp_values = [
            as_int(row.get("WPNo"), -1)
            for _, row in indexed_rows
            if as_int(row.get("WPNo"), -1) > 0
        ]

        wp_no = wp_values[0] if wp_values else None

        ono, opos, step_no = key

        decisions.append({
            "run_id": run_id,
            "WPNo": wp_no,
            "StepNo": step_no,
            "ONo": ono,
            "OPos": opos,
            "decision_timestamp": decision_timestamp,
            "MainONo": main_order if main_order > 0 else None,
            "required_PNo": required_pno,
            "selected_BoxID": source_box,
            "selected_BoxPos": source_pos,
        })

    decisions.sort(
        key=lambda row: row["decision_timestamp"]
    )

    for number, decision in enumerate(decisions, start=1):

        decision["decision_id"] = (
            f"{run_id}_{number:08d}"
        )

        decision["decision_timestamp"] = (
            decision["decision_timestamp"].isoformat()
        )

    return decisions, rejected_operations


# ============================================================
# INITIAL STORAGE STATE
# ============================================================

def build_initial_storage_state(
    buffer_rows: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:

    storage_state = {}

    for row in buffer_rows:

        if str(row.get("action", "")).lower() != "initial_state":
            continue

        if as_int(row.get("ResourceId")) != STORAGE_RESOURCE_ID:
            continue

        if as_int(row.get("BufNo")) != STORAGE_BUF_NO:
            continue

        if as_int(row.get("PNo")) != BOX_TYPE_PNO:
            continue

        buf_pos = as_int(row.get("BufPos"), -1)
        box_id = as_int(row.get("BoxID"))

        if buf_pos < 0:
            continue

        if not MIN_BOX_ID <= box_id <= MAX_BOX_ID:
            continue

        # A physical box can occupy only one Storage position.
        for old_key, old_state in list(storage_state.items()):
            if old_state["BoxID"] == box_id:
                del storage_state[old_key]

        # One box per Storage position.
        storage_state[(STORAGE_BUF_NO, buf_pos)] = {
            "BoxID": box_id,
            "PNoGroup": as_int(row.get("PNoGroup")),
        }

    return storage_state


# ============================================================
# INITIAL BOX CONTENTS
# ============================================================

def build_initial_box_state(
    box_rows: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:

    box_state = {}

    for row in box_rows:

        if str(row.get("action", "")).lower() != "initial_state":
            continue

        box_id = as_int(row.get("BoxId"))
        box_pos = as_int(row.get("BoxPos"), -1)

        if box_id <= 0:
            continue

        if box_pos not in BOX_POSITIONS:
            continue

        box_state[(box_id, box_pos)] = row

    return box_state


# ============================================================
# BUILD HISTORICAL STATE EVENTS
# ============================================================

def build_state_events(
    buffer_rows: list[dict[str, Any]],
    box_rows: list[dict[str, Any]],
) -> list[tuple[datetime, int, str, dict[str, Any]]]:

    events = []
    sequence = 0

    # Storage and Pick & Sort are the only buffer locations
    # needed to reconstruct Storage availability.
    for row in buffer_rows:

        action = str(row.get("action", "")).lower()

        if action == "initial_state":
            continue

        if as_int(row.get("ResourceId")) != STORAGE_RESOURCE_ID:
            continue

        buf_no = as_int(row.get("BufNo"))

        if buf_no not in {
            STORAGE_BUF_NO,
            PICK_SORT_BUF_NO,
        }:
            continue

        if as_int(row.get("PNo")) != BOX_TYPE_PNO:
            continue

        timestamp = get_timestamp(row)

        if timestamp is None:
            continue

        events.append((
            timestamp,
            sequence,
            "buffer",
            row,
        ))

        sequence += 1

    # Track box contents independently of buffer movements.
    for row in box_rows:

        if str(row.get("action", "")).lower() == "initial_state":
            continue

        timestamp = get_timestamp(row)

        if timestamp is None:
            continue

        events.append((
            timestamp,
            sequence,
            "box",
            row,
        ))

        sequence += 1

    events.sort(
        key=lambda event: (
            event[0],
            event[1],
        )
    )

    return events


# ============================================================
# APPLY ONE STATE EVENT
# ============================================================

def apply_state_event(
    event_type: str,
    row: dict[str, Any],
    buffer_state: dict[tuple[int, int], dict[str, Any]],
    box_state: dict[tuple[int, int], dict[str, Any]],
) -> None:

    action = str(row.get("action", "")).lower()

    if event_type == "buffer":

        buf_no = as_int(row.get("BufNo"))
        buf_pos = as_int(row.get("BufPos"), -1)
        box_id = as_int(row.get("BoxID"))

        if not MIN_BOX_ID <= box_id <= MAX_BOX_ID:
            return

        # The box has left Storage when it reaches Pick & Sort.
        if buf_no == PICK_SORT_BUF_NO:

            for old_key, old_state in list(buffer_state.items()):

                if old_state["BoxID"] == box_id:
                    del buffer_state[old_key]

            return

        if buf_no != STORAGE_BUF_NO:
            return

        key = (
            STORAGE_BUF_NO,
            buf_pos,
        )

        if action == "delete":

            current = buffer_state.get(key)

            if (
                current is not None
                and current["BoxID"] == box_id
            ):
                buffer_state.pop(key, None)

            return

        if action not in {"insert", "update"}:
            return

        if buf_pos < 0:
            return

        # Returning to Storage makes the box available again,
        # regardless of the buffer from which it returned.
        for old_key, old_state in list(buffer_state.items()):

            if old_state["BoxID"] == box_id:
                del buffer_state[old_key]

        # Remove any previous occupant of this Storage position.
        buffer_state.pop(key, None)

        buffer_state[key] = {
            "BoxID": box_id,
            "PNoGroup": as_int(row.get("PNoGroup")),
        }

        return

    if event_type == "box":

        box_id = as_int(row.get("BoxId"))
        box_pos = as_int(row.get("BoxPos"), -1)

        if box_id <= 0:
            return

        if box_pos not in BOX_POSITIONS:
            return

        key = (
            box_id,
            box_pos,
        )

        if action == "delete":
            box_state.pop(key, None)

        elif action in {"insert", "update"}:
            box_state[key] = row


# ============================================================
# GET BOXES CURRENTLY IN STORAGE
# ============================================================

def get_available_storage_boxes(
    buffer_state: dict[tuple[int, int], dict[str, Any]],
) -> dict[int, int]:

    available_boxes = {}

    for (buf_no, buf_pos), state in buffer_state.items():

        if buf_no != STORAGE_BUF_NO:
            continue

        box_id = as_int(state.get("BoxID"))

        if box_id > 0:
            available_boxes[box_id] = buf_pos

    return available_boxes


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    decision: dict[str, Any],
    available_boxes: dict[int, int],
    box_state: dict[tuple[int, int], dict[str, Any]],
    order_hierarchy: dict[int, int],
) -> list[dict[str, Any]]:

    required_pno = as_int(
        decision["required_PNo"]
    )

    candidates = []

    for box_id, buf_pos in sorted(
        available_boxes.items()
    ):

        box_features = build_box_features(
            box_id,
            box_state,
            order_hierarchy,
        )

        for box_pos in BOX_POSITIONS:

            record = box_state.get(
                (box_id, box_pos),
                {},
            )

            pno = as_int(record.get("PNo"))
            ono = as_int(record.get("ONo"))
            opos = as_int(record.get("OPos"))

            is_occupied = pno != 0
            is_required_part = pno == required_pno

            candidates.append({

                "decision_id": decision["decision_id"],
                "run_id": decision["run_id"],
                "decision_timestamp": decision["decision_timestamp"],

                "required_PNo": required_pno,
                "MainONo": decision["MainONo"],

                "BoxID": box_id,
                "BoxPos": box_pos,
                "storage_BufPos": buf_pos,

                "PNo": pno,
                "ONo": ono,
                "OPos": opos,

                "is_occupied": is_occupied,
                "contains_required_PNo": is_required_part,
                "is_feasible_action": is_required_part,

                **box_features,

                "selected": (
                    box_id == decision["selected_BoxID"]
                    and box_pos == decision["selected_BoxPos"]
                ),
            })

    return candidates


# ============================================================
# STORAGE SNAPSHOT
# ============================================================

def build_storage_snapshot(
    decision: dict[str, Any],
    buffer_state: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:

    snapshots = []

    for (buf_no, buf_pos), state in sorted(
        buffer_state.items()
    ):

        if buf_no != STORAGE_BUF_NO:
            continue

        box_id = as_int(state.get("BoxID"))

        snapshots.append({

            "decision_id": decision["decision_id"],
            "run_id": decision["run_id"],
            "decision_timestamp": decision["decision_timestamp"],

            "BufPos": buf_pos,
            "BoxID": box_id,
            "occupied": box_id > 0,
        })

    return snapshots


# ============================================================
# VALIDATE RECONSTRUCTED STATE
# ============================================================

def validate_decision_state(
    decision: dict[str, Any],
    available_boxes: dict[int, int],
    candidates: list[dict[str, Any]],
) -> None:

    positions = list(available_boxes.values())

    if len(positions) != len(set(positions)):
        raise AssertionError(
            f"Duplicate Storage BufPos at {decision['decision_id']}"
        )

    if len(available_boxes) != len(set(available_boxes.keys())):
        raise AssertionError(
            f"Duplicate BoxID in Storage at {decision['decision_id']}"
        )

    expected_rows = (
        len(available_boxes) * len(BOX_POSITIONS)
    )

    if len(candidates) != expected_rows:
        raise AssertionError(
            f"Candidate row count mismatch at "
            f"{decision['decision_id']}: "
            f"expected {expected_rows}, got {len(candidates)}"
        )

    candidate_keys = [
        (c["BoxID"], c["BoxPos"])
        for c in candidates
    ]

    if len(candidate_keys) != len(set(candidate_keys)):
        raise AssertionError(
            f"Duplicate candidate BoxID/BoxPos at "
            f"{decision['decision_id']}"
        )


# ============================================================
# CLASSIFY SELECTED ACTION
# ============================================================

def classify_selected_action(
    decision: dict[str, Any],
    available_boxes: dict[int, int],
    candidates: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:

    selected_box = as_int(
        decision["selected_BoxID"]
    )

    selected_pos = as_int(
        decision["selected_BoxPos"]
    )

    if not MIN_BOX_ID <= selected_box <= MAX_BOX_ID:
        return "selected_box_id_out_of_range", None

    if selected_pos not in BOX_POSITIONS:
        return "selected_position_out_of_range", None

    if selected_box not in available_boxes:
        return "selected_box_not_in_storage", None

    selected_rows = [
        c for c in candidates
        if (
            c["BoxID"] == selected_box
            and c["BoxPos"] == selected_pos
        )
    ]

    if len(selected_rows) != 1:
        return "selected_position_not_found", None

    selected_row = selected_rows[0]

    if not selected_row["is_occupied"]:
        return "selected_position_empty", selected_row

    if selected_row["PNo"] != decision["required_PNo"]:
        return "selected_position_wrong_part", selected_row

    return "valid", selected_row


# ============================================================
# PROCESS ONE RUN
# ============================================================

def process_run(run_id: str):

    print(f"\nProcessing run {run_id}...")

    buffer_df = load_table(
        run_id,
        "tblbufferpos",
    )

    box_df = load_table(
        run_id,
        "tblboxpos",
    )

    step_df = load_table(
        run_id,
        "tblstepparameter",
    )

    validate_columns(
        run_id,
        "tblbufferpos",
        buffer_df,
        {
            "timestamp",
            "action",
            "ResourceId",
            "BufNo",
            "BufPos",
            "BoxID",
            "PNo",
        },
    )

    validate_columns(
        run_id,
        "tblboxpos",
        box_df,
        {
            "timestamp",
            "action",
            "BoxPos",
            "BoxId",
            "PNo",
            "ONo",
            "OPos",
        },
    )

    validate_columns(
        run_id,
        "tblstepparameter",
        step_df,
        {
            "timestamp",
            "action",
            "WPNo",
            "StepNo",
            "ONo",
            "OPos",
            "ParameterNo",
            "Parameter",
        },
    )

    # --------------------------------------------------------
    # 1. Build suborder -> main order hierarchy.
    #    ParameterNo 8 is optional metadata.
    # --------------------------------------------------------

    hierarchy_rows = (
        step_df
        .filter(
            (
                pl.col("action")
                .cast(pl.Utf8)
                .str.to_lowercase()
                .is_in(["insert", "update"])
            )
            & (
                pl.col("StepNo")
                .cast(pl.Int64, strict=False)
                == EXTRACTION_STEP_NO
            )
            & (
                pl.col("ParameterNo")
                .cast(pl.Int64, strict=False)
                == PARAM_MAIN_ORDER
            )
            & (
                pl.col("ONo")
                .cast(pl.Int64, strict=False)
                > 1
            )
        )
        .select([
            pl.col("ONo")
            .cast(pl.Int64, strict=False)
            .alias("SubOrderNo"),

            pl.col("Parameter")
            .cast(pl.Int64, strict=False)
            .alias("MainONo"),
        ])
        .drop_nulls()
        .unique()
    )

    order_hierarchy = {
        as_int(row["SubOrderNo"]): as_int(row["MainONo"])
        for row in hierarchy_rows.to_dicts()
        if (
            as_int(row["SubOrderNo"]) > 0
            and as_int(row["MainONo"]) > 0
        )
    }

    # --------------------------------------------------------
    # 2. Reconstruct MES decisions.
    # --------------------------------------------------------

    decisions, rejected_operations = (
        build_extraction_operations(
            step_df,
            run_id,
        )
    )

    print(
        f"  Logical MES decisions: {len(decisions):,}"
    )

    print(
        f"  Rejected malformed operations: "
        f"{rejected_operations:,}"
    )

    # --------------------------------------------------------
    # 3. Initialize physical Storage and box contents.
    # --------------------------------------------------------

    buffer_rows = buffer_df.to_dicts()
    box_rows = box_df.to_dicts()

    buffer_state = build_initial_storage_state(
        buffer_rows
    )

    box_state = build_initial_box_state(
        box_rows
    )

    events = build_state_events(
        buffer_rows,
        box_rows,
    )

    all_candidates = []
    all_storage_snapshots = []

    event_index = 0

    # --------------------------------------------------------
    # 4. Replay events before each MES decision.
    # --------------------------------------------------------

    for decision in decisions:

        decision_ts = as_datetime(
            decision["decision_timestamp"]
        )

        if decision_ts is None:
            continue

        # Strictly before the first source-selection write.
        while (
            event_index < len(events)
            and events[event_index][0] < decision_ts
        ):

            _, _, event_type, row = events[event_index]

            apply_state_event(
                event_type,
                row,
                buffer_state,
                box_state,
            )

            event_index += 1

        available_boxes = get_available_storage_boxes(
            buffer_state
        )

        # ----------------------------------------------------
        # 5. Generate candidates from the Storage snapshot.
        # ----------------------------------------------------

        candidates = generate_candidates(
            decision,
            available_boxes,
            box_state,
            order_hierarchy,
        )

        validate_decision_state(
            decision,
            available_boxes,
            candidates,
        )

        selected_status, selected_row = (
            classify_selected_action(
                decision,
                available_boxes,
                candidates,
            )
        )

        feasible_count = sum(
            c["is_feasible_action"]
            for c in candidates
        )

        occupied_count = sum(
            c["is_occupied"]
            for c in candidates
        )

        decision["state_position_count"] = len(candidates)

        decision["storage_box_count"] = len(
            available_boxes
        )

        decision["candidate_count"] = feasible_count

        decision["occupied_position_count"] = (
            occupied_count
        )

        decision["selected_box_in_storage"] = (
            decision["selected_BoxID"] in available_boxes
        )

        decision["selected_candidate_exists"] = (
            selected_row is not None
        )

        decision["selected_position_PNo"] = (
            selected_row["PNo"]
            if selected_row is not None
            else None
        )

        decision["selected_position_occupied"] = (
            selected_row["is_occupied"]
            if selected_row is not None
            else False
        )

        decision["selected_position_matches_required_PNo"] = (
            selected_row is not None
            and selected_row["PNo"] == decision["required_PNo"]
        )

        decision["selected_storage_BufPos"] = (
            available_boxes.get(
                decision["selected_BoxID"]
            )
        )

        decision["validation_status"] = selected_status

        decision["valid_for_training"] = (
            selected_status == "valid"
        )

        for candidate in candidates:
            candidate["valid_for_training"] = (
                decision["valid_for_training"]
            )

        all_candidates.extend(candidates)

        all_storage_snapshots.extend(
            build_storage_snapshot(
                decision,
                buffer_state,
            )
        )

    valid_count = sum(
        d["valid_for_training"]
        for d in decisions
    )

    invalid_count = len(decisions) - valid_count

    print(
        f"  Candidate rows: {len(all_candidates):,}"
    )

    print(
        f"  Storage snapshot rows: "
        f"{len(all_storage_snapshots):,}"
    )

    print(
        f"  Valid decisions: {valid_count:,}"
    )

    print(
        f"  Invalid decisions: {invalid_count:,}"
    )

    return (
        decisions,
        all_candidates,
        all_storage_snapshots,
    )


# ============================================================
# PROCESS ALL RUNS
# ============================================================

all_decisions = []
all_candidates = []
all_storage = []

for run in RUN_IDS:

    decisions, candidates, storage = process_run(run)

    all_decisions.extend(decisions)
    all_candidates.extend(candidates)
    all_storage.extend(storage)


# ============================================================
# CREATE DATAFRAMES
# ============================================================

df_decisions = (
    pl.DataFrame(all_decisions)
    if all_decisions
    else pl.DataFrame()
)

df_candidates = (
    pl.DataFrame(all_candidates)
    if all_candidates
    else pl.DataFrame()
)

df_storage_state = (
    pl.DataFrame(all_storage)
    if all_storage
    else pl.DataFrame()
)


# ============================================================
# ADD BOX COMPOSITION FEATURES
# ============================================================

if not df_candidates.is_empty():

    df_candidates = add_box_composition_features(
        df_candidates
    )


# ============================================================
# FINAL DATASET INTEGRITY CHECKS
# ============================================================

if not df_decisions.is_empty():

    # Unique decision IDs.
    if (
        df_decisions
        .get_column("decision_id")
        .n_unique()
        != df_decisions.height
    ):
        raise AssertionError(
            "Duplicate decision_id values"
        )

    # Candidate rows = 8 positions per Storage box.
    candidate_counts = (
        df_candidates
        .group_by("decision_id")
        .agg(
            pl.len().alias("rows")
        )
        if not df_candidates.is_empty()
        else pl.DataFrame({
            "decision_id": [],
            "rows": [],
        })
    )

    expected_counts = df_decisions.select([
        "decision_id",

        (
            pl.col("storage_box_count")
            * len(BOX_POSITIONS)
        ).alias("expected_rows"),
    ])

    count_check = (
        expected_counts
        .join(
            candidate_counts,
            on="decision_id",
            how="left",
        )
        .with_columns(
            pl.col("rows").fill_null(0)
        )
    )

    bad_counts = count_check.filter(
        pl.col("expected_rows") != pl.col("rows")
    )

    if not bad_counts.is_empty():

        raise AssertionError(
            "Candidate row integrity failure:\n"
            f"{bad_counts}"
        )

    # At most one selected source row per decision.
    if not df_candidates.is_empty():

        selected_counts = (
            df_candidates
            .filter(pl.col("selected"))
            .group_by("decision_id")
            .agg(
                pl.len().alias("selected_rows")
            )
        )

        if not selected_counts.is_empty():

            duplicates = selected_counts.filter(
                pl.col("selected_rows") > 1
            )

            if not duplicates.is_empty():

                raise AssertionError(
                    "More than one selected candidate "
                    f"per decision:\n{duplicates}"
                )


# ============================================================
# SAVE DATASETS
# ============================================================

df_decisions.write_parquet(
    OUTPUT_DIR / "decisions.parquet"
)

df_candidates.write_parquet(
    OUTPUT_DIR / "candidates.parquet"
)

df_storage_state.write_parquet(
    OUTPUT_DIR / "storage_state.parquet"
)


# ============================================================
# VALIDATION SUMMARY
# ============================================================

print("\nDatasets created:")

print(
    f"  decisions.parquet       "
    f"{df_decisions.height:>10,} rows"
)

print(
    f"  candidates.parquet      "
    f"{df_candidates.height:>10,} rows"
)

print(
    f"  storage_state.parquet   "
    f"{df_storage_state.height:>10,} rows"
)


if not df_decisions.is_empty():

    valid_count = (
        df_decisions
        .filter(pl.col("valid_for_training"))
        .height
    )

    invalid_count = (
        df_decisions.height - valid_count
    )

    print("\nDecision quality:")

    print(
        f"  Valid for training:   {valid_count:,}"
    )

    print(
        f"  Invalid for training: {invalid_count:,}"
    )

    print("\nValidation status:")

    print(
        df_decisions
        .group_by("validation_status")
        .agg(
            pl.len().alias("decisions")
        )
        .sort("validation_status")
    )

    print("\nDecisions by run:")

    print(
        df_decisions
        .group_by("run_id")
        .agg([
            pl.len().alias("decisions"),

            pl.col("valid_for_training")
            .sum()
            .alias("valid_decisions"),

            pl.col("candidate_count")
            .mean()
            .round(2)
            .alias("mean_feasible_candidates"),
        ])
        .sort("run_id")
    )

    print("\nDecisions by demanded part:")

    print(
        df_decisions
        .group_by("required_PNo")
        .agg([
            pl.len().alias("decisions"),

            pl.col("valid_for_training")
            .sum()
            .alias("valid_decisions"),
        ])
        .sort("required_PNo")
    )

    print("\nDecisions requiring attention:")

    print(
        df_decisions
        .filter(~pl.col("valid_for_training"))
        .select([
            "decision_id",
            "run_id",
            "decision_timestamp",
            "WPNo",
            "StepNo",
            "ONo",
            "OPos",
            "MainONo",
            "required_PNo",
            "selected_BoxID",
            "selected_BoxPos",
            "selected_box_in_storage",
            "selected_position_PNo",
            "selected_position_occupied",
            "validation_status",
            "candidate_count",
            "state_position_count",
        ])
    )


if not df_candidates.is_empty():

    print("\nCandidate position state:")

    print(
        df_candidates
        .group_by([
            "is_occupied",
            "contains_required_PNo",
        ])
        .agg(
            pl.len().alias("rows")
        )
        .sort([
            "is_occupied",
            "contains_required_PNo",
        ])
    )


print(f"\nOutput directory: {OUTPUT_DIR}")