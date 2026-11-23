//! Contract test for `docs/architecture/TURN-CONTRACT.md`.
//!
//! The document is hand-written; this test is the mechanical check. It parses
//! the states, legal transitions, and floor-ownership tables out of the
//! markdown and compares them with the code's [`LEGAL_TRANSITIONS`],
//! [`ALL_STATES`], and [`TurnState::owns_floor`]. Any drift fails with the
//! offending rows named.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::PathBuf;

use parlevox_core::{
    ParlevoxError, TurnEventKind, TurnMachine, TurnState, ALL_EVENT_KINDS, ALL_STATES,
    LEGAL_TRANSITIONS,
};

/// Path to the checked-in prose contract.
fn contract_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/architecture/TURN-CONTRACT.md")
}

/// Read the prose contract, failing loudly if it is missing.
fn read_contract() -> String {
    let path = contract_path();
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}

/// Extract one `## <heading>` section, up to the next level-two heading.
fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let marker = format!("## {heading}");
    let start = doc
        .find(marker.as_str())
        .unwrap_or_else(|| panic!("TURN-CONTRACT.md is missing the `## {heading}` section"));
    let rest = &doc[start + marker.len()..];
    match rest.find("\n## ") {
        Some(end) => &rest[..end],
        None => rest,
    }
}

/// A markdown table cell that is purely the separator row (`---`, `:---:`).
fn is_separator_cell(cell: &str) -> bool {
    !cell.is_empty() && cell.chars().all(|character| matches!(character, '-' | ':'))
}

/// Body rows of the first markdown table in `section`.
///
/// The header row and the `---` separator are skipped; a section with no table
/// yields an empty vector, which the callers assert against.
fn table_rows(section: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut saw_separator = false;

    for line in section.lines() {
        let trimmed = line.trim();
        if !(trimmed.starts_with('|') && trimmed.ends_with('|')) {
            continue;
        }
        let cells: Vec<String> = trimmed
            .trim_matches('|')
            .split('|')
            .map(|cell| cell.trim().to_string())
            .collect();
        if cells.iter().all(|cell| is_separator_cell(cell)) {
            saw_separator = true;
            continue;
        }
        if !saw_separator {
            // Header row, not a data row.
            continue;
        }
        rows.push(cells);
    }

    rows
}

/// Strip the backticks and padding the document uses for canonical names.
fn canonical(cell: &str) -> &str {
    cell.trim().trim_matches('`').trim()
}

/// Canonical state name to variant.
fn state_by_name() -> BTreeMap<&'static str, TurnState> {
    ALL_STATES
        .iter()
        .map(|state| (state.name(), *state))
        .collect()
}

/// Canonical event-kind name to variant.
fn event_kind_by_name() -> BTreeMap<&'static str, TurnEventKind> {
    ALL_EVENT_KINDS
        .iter()
        .map(|kind| (kind.name(), *kind))
        .collect()
}

fn resolve_state(by_name: &BTreeMap<&'static str, TurnState>, cell: &str) -> TurnState {
    let name = canonical(cell);
    *by_name
        .get(name)
        .unwrap_or_else(|| panic!("document names unknown state `{name}`"))
}

fn resolve_event_kind(
    by_name: &BTreeMap<&'static str, TurnEventKind>,
    cell: &str,
) -> TurnEventKind {
    let name = canonical(cell);
    *by_name
        .get(name)
        .unwrap_or_else(|| panic!("document names unknown event `{name}`"))
}

/// A legal path from `Idle` to each state, derived only from the table.
///
/// Using the table itself keeps the driver honest: a state can only be reached
/// by events the machine actually accepts, never by poking a private field.
fn reachable_paths() -> BTreeMap<TurnState, Vec<TurnEventKind>> {
    let mut paths = BTreeMap::new();
    paths.insert(TurnState::Idle, Vec::new());
    let mut frontier = VecDeque::from([TurnState::Idle]);

    while let Some(state) = frontier.pop_front() {
        let path = paths
            .get(&state)
            .expect("every queued state has a recorded path")
            .clone();
        for &(from, kind, to) in LEGAL_TRANSITIONS {
            if from != state || paths.contains_key(&to) {
                continue;
            }
            let mut next = path.clone();
            next.push(kind);
            paths.insert(to, next);
            frontier.push_back(to);
        }
    }

    paths
}

/// Drive a fresh machine into `state`, or `None` if the table cannot reach it.
fn machine_in(
    state: TurnState,
    paths: &BTreeMap<TurnState, Vec<TurnEventKind>>,
) -> Option<TurnMachine> {
    let path = paths.get(&state)?;
    let mut machine = TurnMachine::new();
    for &kind in path {
        machine
            .apply(kind.sample_event())
            .expect("a table-derived path must be accepted by apply");
    }
    assert_eq!(
        machine.state(),
        state,
        "table-derived path must land in {state:?}"
    );
    Some(machine)
}

#[test]
fn documented_states_match_the_code() {
    // Arrange
    let doc = read_contract();
    let state_names = state_by_name();
    let rows = table_rows(section(&doc, "States"));
    assert!(!rows.is_empty(), "the `## States` table has no rows");

    // Act
    let documented: BTreeSet<TurnState> = rows
        .iter()
        .map(|row| resolve_state(&state_names, &row[0]))
        .collect();
    let coded: BTreeSet<TurnState> = ALL_STATES.iter().copied().collect();

    // Assert
    let missing_from_doc: Vec<&str> = coded.difference(&documented).map(|s| s.name()).collect();
    let extra_in_doc: Vec<&str> = documented.difference(&coded).map(|s| s.name()).collect();
    assert!(
        missing_from_doc.is_empty() && extra_in_doc.is_empty(),
        "states differ: missing from doc {missing_from_doc:?}, extra in doc {extra_in_doc:?}"
    );
}

#[test]
fn documented_legal_transitions_match_the_code() {
    // Arrange
    let doc = read_contract();
    let state_names = state_by_name();
    let event_names = event_kind_by_name();
    let rows = table_rows(section(&doc, "Legal transitions"));
    assert!(
        !rows.is_empty(),
        "the `## Legal transitions` table has no rows"
    );

    // Act
    let documented: BTreeSet<(TurnState, TurnEventKind, TurnState)> = rows
        .iter()
        .map(|row| {
            assert_eq!(
                row.len(),
                3,
                "a legal transition row must have three cells: {row:?}"
            );
            (
                resolve_state(&state_names, &row[0]),
                resolve_event_kind(&event_names, &row[1]),
                resolve_state(&state_names, &row[2]),
            )
        })
        .collect();
    let coded: BTreeSet<(TurnState, TurnEventKind, TurnState)> =
        LEGAL_TRANSITIONS.iter().copied().collect();

    // Assert
    let missing_from_doc: Vec<(TurnState, TurnEventKind, TurnState)> =
        coded.difference(&documented).copied().collect();
    let extra_in_doc: Vec<(TurnState, TurnEventKind, TurnState)> =
        documented.difference(&coded).copied().collect();
    assert!(
        missing_from_doc.is_empty() && extra_in_doc.is_empty(),
        "legal transitions differ: present in code but absent from doc {missing_from_doc:?}; \
         documented but not legal in code {extra_in_doc:?}"
    );
}

#[test]
fn floor_ownership_matches_the_documented_contract() {
    // Arrange
    let doc = read_contract();
    let state_names = state_by_name();
    let rows = table_rows(section(&doc, "Floor ownership"));
    assert!(
        !rows.is_empty(),
        "the `## Floor ownership` table has no rows"
    );

    // Act
    let mut documented: BTreeMap<TurnState, String> = BTreeMap::new();
    for row in &rows {
        assert!(
            row.len() >= 2,
            "a floor-ownership row must have at least two cells: {row:?}"
        );
        documented.insert(
            resolve_state(&state_names, &row[0]),
            canonical(&row[1]).to_string(),
        );
    }

    // Assert
    for &state in ALL_STATES {
        let documented_owner = documented
            .get(&state)
            .unwrap_or_else(|| panic!("floor-ownership table does not list {state:?}"));
        assert_eq!(
            documented_owner,
            state.owns_floor().label(),
            "floor owner for {state:?} disagrees with the code"
        );
    }

    let coded: BTreeSet<TurnState> = ALL_STATES.iter().copied().collect();
    let documented_states: BTreeSet<TurnState> = documented.keys().copied().collect();
    assert_eq!(
        documented_states, coded,
        "the floor-ownership table must list every state exactly once"
    );
}

#[test]
fn illegal_transitions_return_the_documented_error_classes() {
    // Arrange
    let paths = reachable_paths();
    let legal_target: BTreeMap<(TurnState, TurnEventKind), TurnState> = LEGAL_TRANSITIONS
        .iter()
        .map(|&(from, kind, to)| ((from, kind), to))
        .collect();

    let mut checked = 0_u32;
    let mut skipped = 0_u32;

    // Act + Assert
    for &state in ALL_STATES {
        if !paths.contains_key(&state) {
            println!("skipping state {state:?}: unreachable from Idle through the legal table");
            skipped += 1;
            continue;
        }
        for &kind in ALL_EVENT_KINDS {
            let mut machine = machine_in(state, &paths)
                .unwrap_or_else(|| panic!("state {state:?} was reachable above"));
            let before_state = machine.state();
            let before_turn = machine.turn_index();
            let event = kind.sample_event();
            let outcome = machine.apply(event.clone());
            checked += 1;

            if state == TurnState::Closed {
                assert!(
                    matches!(&outcome, Err(ParlevoxError::Closed)),
                    "Closed + {} must return ParlevoxError::Closed, got {outcome:?}",
                    kind.name()
                );
                assert_eq!(
                    machine.state(),
                    TurnState::Closed,
                    "a rejected event must not change the state"
                );
                assert_eq!(
                    machine.turn_index(),
                    before_turn,
                    "a rejected event must not change the turn index"
                );
                continue;
            }

            match legal_target.get(&(state, kind)) {
                Some(&target) => {
                    let transition = outcome.as_ref().unwrap_or_else(|error| {
                        panic!(
                            "legal row {state:?} + {} must be accepted, got {error:?}",
                            kind.name()
                        )
                    });
                    assert_eq!(transition.from, before_state, "transition.from disagrees");
                    assert_eq!(
                        transition.to,
                        target,
                        "legal row {state:?} + {} must enter {target:?}",
                        kind.name()
                    );
                    assert_eq!(
                        machine.state(),
                        target,
                        "machine state after accepted event"
                    );
                }
                None => {
                    match &outcome {
                        Err(ParlevoxError::Protocol(message)) => {
                            let expected =
                                format!("illegal turn transition: {state:?} + {event:?}");
                            assert_eq!(
                                message,
                                &expected,
                                "illegal pair {state:?} + {} named the wrong event class",
                                kind.name()
                            );
                        }
                        other => panic!(
                            "expected ParlevoxError::Protocol for {state:?} + {}, got {other:?}",
                            kind.name()
                        ),
                    }
                    assert_eq!(
                        machine.state(),
                        before_state,
                        "a rejected event must not change the state"
                    );
                    assert_eq!(
                        machine.turn_index(),
                        before_turn,
                        "a rejected event must not change the turn index"
                    );
                }
            }
        }
    }

    let total = u32::try_from(ALL_STATES.len() * ALL_EVENT_KINDS.len())
        .expect("the state/event product fits in u32");
    assert_eq!(
        checked + skipped * u32::try_from(ALL_EVENT_KINDS.len()).expect("kind count fits"),
        total,
        "the exhaustive driver must account for every (state, event) pair"
    );
    println!("checked {checked} (state, event) pairs across {skipped} skipped states");
}

#[test]
fn only_final_transcript_advances_the_turn_index() {
    // Arrange
    let paths = reachable_paths();
    let mut checked = 0_usize;

    // Act + Assert
    for &(from, kind, to) in LEGAL_TRANSITIONS {
        if !paths.contains_key(&from) {
            println!("skipping transition from {from:?}: unreachable from Idle");
            continue;
        }
        let mut machine = machine_in(from, &paths)
            .unwrap_or_else(|| panic!("state {from:?} was reachable above"));
        let before = machine.turn_index();
        let transition = machine.apply(kind.sample_event()).unwrap_or_else(|error| {
            panic!(
                "legal row {from:?} + {} must be accepted, got {error:?}",
                kind.name()
            )
        });

        assert_eq!(transition.from, from, "transition.from disagrees");
        assert_eq!(transition.to, to, "transition.to disagrees with the table");
        let expected = if kind == TurnEventKind::FinalTranscript {
            before + 1
        } else {
            before
        };
        assert_eq!(
            transition.turn,
            expected,
            "{from:?} + {} produced the wrong turn delta",
            kind.name()
        );
        assert_eq!(
            machine.turn_index(),
            expected,
            "the machine's turn index must match the transition"
        );
        checked += 1;
    }

    assert_eq!(
        checked,
        LEGAL_TRANSITIONS.len(),
        "every legal transition must be exercised"
    );
    println!("checked {checked} legal transitions for the turn-index rule");
}
