"""
Streamlit UI for the agentic mRNA design pipeline.

Launch with::

    streamlit run ui/app.py

Four tabs:

* **Run**         configure and launch an optimisation, watch it converge live
* **Pareto front** explore the resulting trade-off menu and export sequences
* **Compare runs** read back every run that has written a manifest
* **Environment**  which optional tools are present, and what is silently missing

The optimisation itself runs on a worker thread (see ``ui/runner.py``); this
module only renders state.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from mrna_design.controllers import CONTROLLER_CHOICES, CONTROLLER_DESCRIPTIONS
from mrna_design.metrics.objective_spec import OBJECTIVE_SETS, get_objective_set
from ui.runner import (
    RunConfig,
    archive_rows,
    candidate_by_id,
    discover_runs,
    start_run,
    to_fasta,
    validate_protein,
)


# `use_container_width` is deprecated in current Streamlit in favour of `width`,
# but `width` does not exist on older releases. Detect once and spread the right
# keyword, so the app works across both.
def _full_width() -> dict:
    import inspect

    try:
        params = inspect.signature(st.dataframe).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return {"use_container_width": True}
    return {"width": "stretch"} if "width" in params else {"use_container_width": True}


FULL_WIDTH = _full_width()

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGETS_FILE = REPO_ROOT / "data" / "targets" / "benchmark_proteins.json"
RESULTS_ROOT = REPO_ROOT / "results"
REFRESH_SECONDS = 2.0

st.set_page_config(page_title="mRNA Design", page_icon="\U0001f9ec", layout="wide")


# ── Data helpers ──────────────────────────────────────────────────────────────


@st.cache_data
def load_targets() -> list[dict]:
    if not TARGETS_FILE.exists():
        return []
    return json.loads(TARGETS_FILE.read_text()).get("targets", [])


@st.cache_data
def environment_report() -> tuple[list[str], dict]:
    from mrna_design.provenance import environment_manifest, warnings_for_missing_tools

    return warnings_for_missing_tools(), environment_manifest()


def fmt(value, spec: str = "{:.4f}", dash: str = "—") -> str:
    return dash if value is None else spec.format(value)


# ── Sidebar: run configuration ────────────────────────────────────────────────


def sidebar_config() -> RunConfig:
    st.sidebar.header("Run configuration")

    targets = load_targets()
    names = [t["name"] for t in targets] + ["Custom protein"]
    choice = st.sidebar.selectbox(
        "Target protein",
        names,
        help="Benchmark targets ship with the repo. EGFP is the fastest.",
    )

    if choice == "Custom protein":
        protein = (
            st.sidebar.text_area(
                "Amino acid sequence",
                height=120,
                placeholder="MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL...",
                help="Single-letter codes, uppercase. The stop codon is added for you.",
            )
            .strip()
            .upper()
            .replace("\n", "")
            .replace(" ", "")
        )
        utr5 = utr3 = ""
        protein_name = "custom"
    else:
        target = next(t for t in targets if t["name"] == choice)
        protein = target["protein"]
        utr5 = target.get("utr5", "")
        utr3 = target.get("utr3", "")
        protein_name = target["name"]
        st.sidebar.caption(
            f"{len(protein)} aa · CDS {len(protein) * 3 + 3} nt "
            f"· 5'UTR {len(utr5)} nt · 3'UTR {len(utr3)} nt"
        )

    controller = st.sidebar.selectbox(
        "Controller",
        CONTROLLER_CHOICES,
        index=CONTROLLER_CHOICES.index("nsga2"),
    )
    st.sidebar.caption(CONTROLLER_DESCRIPTIONS.get(controller, ""))

    llm_model = "gpt-4o"
    if controller == "llm":
        llm_model = st.sidebar.text_input("Model", value="gpt-4o")
        st.sidebar.warning(
            "Needs an API key in the environment (OPENAI_API_KEY, "
            "ANTHROPIC_API_KEY or GEMINI_API_KEY). Without one the controller "
            "reports itself unavailable and proposes nothing — the run still "
            "completes rather than crashing."
        )

    objective_set = st.sidebar.selectbox(
        "Objective set",
        list(OBJECTIVE_SETS),
        index=list(OBJECTIVE_SETS).index("primary"),
    )
    oset = get_objective_set(objective_set)
    st.sidebar.caption(f"{len(oset)} objectives: {', '.join(oset.labels)}")

    st.sidebar.subheader("Budget")
    max_evaluations = st.sidebar.slider(
        "Evaluation budget",
        min_value=10,
        max_value=2000,
        value=120,
        step=10,
        help="Full objective evaluations. This is the unit of fairness between "
        "controllers — every controller in a comparison gets the same number.",
    )
    population_k = st.sidebar.slider("Population size", 2, 40, 8)
    max_edits = st.sidebar.slider("Max edits per proposal", 1, 20, 5)
    rng_seed = st.sidebar.number_input("Random seed", value=42, step=1)

    st.sidebar.subheader("Metrics")
    run_structure = st.sidebar.checkbox(
        "Fold with ViennaRNA",
        value=True,
        help="The dominant cost, roughly 2 s per evaluation for EGFP. Turning it "
        "off makes runs fast but the MFE objective becomes meaningless.",
    )
    run_safety = st.sidebar.checkbox(
        "Scan miRNA seeds",
        value=True,
        help="Needs the miRBase database (make db). Without it the objective is "
        "constant across every candidate.",
    )

    with st.sidebar.expander("Advanced"):
        empty_policy = st.selectbox(
            "Empty-proposal policy",
            ["noop", "random"],
            help="What the loop does when a controller proposes nothing. Applied "
            "identically to every controller, which is what keeps the "
            "budget comparison fair.",
        )
        include_lineardesign = st.checkbox(
            "LinearDesign seeding",
            value=False,
            help="Silently falls back to cai_max when the binary is not on PATH.",
        )
        max_iters = st.number_input("Iteration cap", value=500, step=50)

    if run_structure and protein:
        est = len(protein) / 239 * 2.0 * max_evaluations
        st.sidebar.info(f"Rough estimate: **{est / 60:.1f} min** at ~2 s per evaluation.")

    return RunConfig(
        protein=protein,
        protein_name=protein_name,
        utr5=utr5,
        utr3=utr3,
        controller=controller,
        objective_set=objective_set,
        max_evaluations=int(max_evaluations),
        max_iters=int(max_iters),
        population_k=int(population_k),
        max_edits_per_iter=int(max_edits),
        rng_seed=int(rng_seed),
        run_structure=run_structure,
        run_safety=run_safety,
        include_lineardesign=include_lineardesign,
        empty_proposal_policy=empty_policy,
        llm_model=llm_model,
        output_dir=RESULTS_ROOT / "ui",
    )


# ── Tab 1: Run ────────────────────────────────────────────────────────────────


def tab_run(cfg: RunConfig) -> None:
    state = st.session_state.get("run")

    left, right = st.columns([3, 1])
    with left:
        st.subheader("Optimisation")
    with right:
        if state is not None and state.is_running:
            if st.button("Stop", type="secondary", **FULL_WIDTH):
                state.request_cancel()
                st.toast("Stopping after the current iteration…")
        else:
            problem = validate_protein(cfg.protein)
            if st.button(
                "Run optimisation", type="primary", disabled=problem is not None, **FULL_WIDTH
            ):
                st.session_state["run"] = start_run(cfg)
                st.rerun()

    # Show why the button is disabled, outside the narrow right-hand column.
    if state is None or not state.is_running:
        problem = validate_protein(cfg.protein)
        if problem:
            st.warning(problem)

    if state is None:
        st.info(
            "Configure the run in the sidebar, then press **Run optimisation**.\n\n"
            "The pipeline searches synonymous codon choices — sequences that encode "
            "exactly the same protein but differ in folding stability, codon usage "
            "and immune visibility. The result is a Pareto front: a menu of "
            "trade-offs rather than a single answer."
        )
        return

    # ── Status line ──
    status_label = {
        "running": "Running",
        "done": "Finished",
        "cancelled": "Stopped",
        "error": "Failed",
    }.get(state.status, state.status)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Status", status_label)
    c2.metric("Evaluations", f"{state.evaluations}/{state.max_evaluations or '∞'}")
    c3.metric("Front size", state.archive_size)
    c4.metric("Hypervolume", fmt(state.hypervolume))
    c5.metric("% of ceiling", f"{state.normalised_hypervolume * 100:.1f}%")

    if state.is_running:
        st.progress(state.progress, text=f"{state.elapsed:.0f}s elapsed")

    if state.status == "error":
        st.error(state.error)
        with st.expander("Traceback"):
            st.code(state.traceback or "", language="text")

    if state.status == "cancelled":
        st.warning("Stopped early. The front found so far is below and still valid.")

    # ── Convergence ──
    if state.hv_trace:
        st.markdown("##### Convergence")
        trace = pd.DataFrame(state.hv_trace)
        chart = trace[["evaluations", "hypervolume"]].set_index("evaluations")
        st.line_chart(chart, height=280)
        st.caption(
            "Evaluations spent, not iterations — the only axis on which controllers "
            "are comparable, because they generate different numbers of candidates "
            "per iteration."
        )

    # ── Completion detail ──
    if state.status in ("done", "cancelled") and state.archive is not None:
        archive = state.archive
        budget = getattr(archive, "run_budget", None)
        stats = getattr(archive, "run_stats", None)

        if budget is not None and stats is not None:
            st.markdown("##### Run accounting")
            a, b, c, d = st.columns(4)
            a.metric("Evaluations charged", budget.spent)
            b.metric(
                "Served from cache",
                budget.served_from_cache,
                help="Repeat evaluations of a sequence already scored. A high "
                "rate means the controller keeps re-deriving the same "
                "candidates, i.e. it has saturated.",
            )
            c.metric(
                "Edit acceptance",
                f"{stats.edit_acceptance_rate * 100:.0f}%",
                help="Share of proposed edits that passed validation. A low "
                "rate for an LLM controller is a direct quality signal.",
            )
            d.metric("Stopped because", stats.stop_reason)

            if budget.served_from_cache > budget.spent:
                st.warning(
                    f"This controller produced only **{budget.spent} distinct "
                    f"sequences** while re-deriving {budget.served_from_cache} it had "
                    f"already seen. It saturated well before its budget ran out."
                )

        degenerate = archive.degenerate_objective_labels()
        if degenerate:
            st.warning(
                "These objectives were **constant across the whole front**, so they "
                f"added a dimension without adding information: {', '.join(degenerate)}. "
                "Usually this means an optional database is missing — see the "
                "Environment tab."
            )

        st.success(f"Results written to `{cfg.output_dir / state.run_id}`")

    # Polling happens in main(), after every tab has rendered. Calling
    # st.rerun() here would abort the script mid-pass, leaving the Pareto
    # front, Compare and Environment tabs blank for the whole duration of a
    # run - which is exactly when a user is most likely to click them.


# ── Tab 2: Pareto front ───────────────────────────────────────────────────────


def tab_front() -> None:
    state = st.session_state.get("run")
    archive = getattr(state, "archive", None) if state else None

    if archive is None or len(archive) == 0:
        st.info("Run an optimisation first; its Pareto front will appear here.")
        return

    rows = archive_rows(archive)
    df = pd.DataFrame(rows)
    display = df.drop(columns=["_sequence_id"])

    st.subheader(f"Pareto front — {len(rows)} non-dominated candidates")
    st.caption(
        "Every row encodes the identical protein. None of them dominates another: "
        "each is better than the rest on at least one objective."
    )
    st.dataframe(display, hide_index=True, **FULL_WIDTH)

    # ── Trade-off scatter ──
    numeric = [
        c
        for c in display.columns
        if pd.api.types.is_numeric_dtype(display[c]) and display[c].notna().any()
    ]
    if len(numeric) >= 2:
        st.markdown("##### Trade-off view")
        c1, c2 = st.columns(2)
        x = c1.selectbox("x-axis", numeric, index=numeric.index("CAI") if "CAI" in numeric else 0)
        y_default = "MFE/nt" if "MFE/nt" in numeric else numeric[-1]
        y = c2.selectbox("y-axis", numeric, index=numeric.index(y_default))
        st.scatter_chart(df, x=x, y=y, height=360)
        st.caption(
            "A front that curves away from the origin is the signature of a real "
            "trade-off: you cannot improve one axis without giving up the other."
        )

    # ── Inspect one candidate ──
    st.markdown("##### Inspect a candidate")
    pick = st.selectbox("Candidate", df["_sequence_id"].tolist(), format_func=lambda s: s[:10])
    cand = candidate_by_id(archive, pick)
    if cand is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CAI", fmt(cand.scores.cai, "{:.3f}"))
        m2.metric("MFE", fmt(cand.scores.mfe, "{:.1f}"))
        m3.metric("GC", fmt(cand.scores.gc_content, "{:.1%}"))
        m4.metric("Accepted edits", len(cand.accepted_edits))

        st.text_area(
            "Full mRNA sequence (5'UTR + CDS + 3'UTR)",
            "\n".join(cand.sequence[i : i + 60] for i in range(0, len(cand.sequence), 60)),
            height=180,
        )

        if cand.accepted_edits:
            with st.expander(f"Edit lineage ({len(cand.accepted_edits)} accepted)"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "codon": r.edit.codon_index,
                                "from": r.edit.original_codon,
                                "to": r.edit.new_codon,
                                "targeting": r.edit.targeting_issue or "—",
                                "reason": r.edit.reason,
                                "iteration": r.iteration,
                            }
                            for r in cand.accepted_edits
                        ]
                    ),
                    hide_index=True,
                    **FULL_WIDTH,
                )

    # ── Export ──
    name = getattr(getattr(state, "config", None), "protein_name", "design")
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download front as FASTA",
        to_fasta(archive, name),
        file_name=f"{name}_pareto_front.fasta",
        mime="text/plain",
        **FULL_WIDTH,
    )
    d2.download_button(
        "Download scores as CSV",
        display.to_csv(index=False),
        file_name=f"{name}_pareto_front.csv",
        mime="text/csv",
        **FULL_WIDTH,
    )


# ── Tab 3: Compare runs ───────────────────────────────────────────────────────


def tab_compare() -> None:
    st.subheader("Previous runs")
    st.caption(
        "Every run writes a manifest recording its objective set, budget, "
        "environment and result. Runs measured in different objective sets are "
        "not comparable with each other."
    )

    runs = discover_runs(RESULTS_ROOT)
    if not runs:
        st.info(
            f"No manifests found under `{RESULTS_ROOT}`. Run something first, "
            f"or point the pipeline here with `--out results/`."
        )
        return

    df = pd.DataFrame(runs)
    st.dataframe(df.drop(columns=["_path"]), hide_index=True, **FULL_WIDTH)

    sets = {r["objectives"] for r in runs if r["objectives"]}
    if len(sets) > 1:
        st.warning(
            f"These runs span different objective sets ({', '.join(sorted(sets))}). "
            "Hypervolumes from different objective spaces are not comparable."
        )
    if any(r["ViennaRNA"] is False for r in runs):
        st.warning(
            "Some runs were made without ViennaRNA. Their structure metrics are "
            "zero and their hypervolumes do not mean what they appear to."
        )

    plottable = df[df["HV"].notna() & df["controller"].notna()]
    if len(plottable) > 1:
        st.markdown("##### Hypervolume by controller")
        st.bar_chart(plottable.groupby("controller")["HV"].mean(), height=300)
        st.caption(
            "Means only. For a defensible comparison use `make bench-full`, "
            "which applies Holm correction and reports effect sizes."
        )


# ── Tab 4: Environment ────────────────────────────────────────────────────────


def tab_environment() -> None:
    warnings, manifest = environment_report()

    st.subheader("Environment")
    st.caption(
        "Every optional tool in this pipeline fails *soft*: without it the run "
        "still produces a hypervolume, it is just not measuring what it claims to. "
        "This tab is how you catch that before it costs you a result."
    )

    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.success("All optional tools and databases are present.")

    tools = manifest.get("external_tools", {})
    st.markdown("##### Tools")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "component": "ViennaRNA (Python bindings)",
                    "available": tools.get("viennarna_python_bindings"),
                    "matters for": (
                        "every structure metric: MFE, ensemble diversity, AUG accessibility"
                    ),
                },
                *[
                    {
                        "component": k,
                        "available": v,
                        "matters for": {
                            "LinearDesign": (
                                "the LinearDesign seed (falls back to cai_max when absent)"
                            ),
                            "blastn": "off-target homology search",
                            "RNAhybrid": "thermodynamic miRNA binding",
                            "LinearFold": "fast folding of sequences over 2500 nt",
                            "RNAfold": "command-line folding (bindings are used instead)",
                            "makeblastdb": "building the BLAST database",
                        }.get(k, ""),
                    }
                    for k, v in tools.get("on_path", {}).items()
                ],
            ]
        ),
        hide_index=True,
        **FULL_WIDTH,
    )

    st.markdown("##### Databases")
    st.dataframe(
        pd.DataFrame(
            [
                {"database": k, "present": v.get("present"), "path": v.get("path")}
                for k, v in manifest.get("databases", {}).items()
            ]
        ),
        hide_index=True,
        **FULL_WIDTH,
    )

    git = manifest.get("git", {})
    st.markdown("##### Code version")
    if git.get("available"):
        cols = st.columns(3)
        cols[0].metric("Commit", (git.get("commit") or "")[:10])
        cols[1].metric("Branch", git.get("branch") or "—")
        cols[2].metric("Uncommitted files", git.get("uncommitted_files", 0))
        if git.get("dirty"):
            st.info(
                "The working tree has uncommitted changes, so runs made now "
                "are not exactly reproducible from the commit alone."
            )
    else:
        st.info("Not a git checkout, so runs cannot record a code version.")

    with st.expander("Package versions"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"package": k, "version": v or "not installed"}
                    for k, v in manifest.get("packages", {}).items()
                ]
            ),
            hide_index=True,
            **FULL_WIDTH,
        )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    st.title("\U0001f9ec Agentic mRNA Design")
    st.caption(
        "Multi-objective synonymous codon optimisation. The controller proposes "
        "edits; a validator guarantees the encoded protein never changes."
    )

    warnings, _ = environment_report()
    if any("ViennaRNA" in w for w in warnings):
        st.error(
            "**ViennaRNA is not installed.** All structure metrics will be zero and "
            "any hypervolume computed over them is meaningless. Install it with "
            "`pip install ViennaRNA`, or untick “Fold with ViennaRNA” to run "
            "without the structure objective."
        )

    cfg = sidebar_config()
    t1, t2, t3, t4 = st.tabs(["Run", "Pareto front", "Compare runs", "Environment"])
    with t1:
        tab_run(cfg)
    with t2:
        tab_front()
    with t3:
        tab_compare()
    with t4:
        tab_environment()

    # Refresh while a run is in flight, once the whole page has been drawn.
    state = st.session_state.get("run")
    if state is not None and state.is_running:
        time.sleep(REFRESH_SECONDS)
        st.rerun()


main()
