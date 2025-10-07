# src/ui/score_renderer.py
"""
render_score utility extracted from original large file.
"""

from src.ui.context import st, pd


def render_score(score_obj, top_k=None):
    if not score_obj:
        st.info("No score to display.")
        return

    ranked = (
        score_obj.get("ranked_candidates")
        or score_obj.get("candidates")
        or score_obj.get("results")
    )
    if ranked and isinstance(ranked, list):
        k = int(top_k or score_obj.get("top_k", 5))
        st.subheader(f"Top {k} candidates (server ranking)")
        rows = []
        for c in ranked[:k]:
            cid = c.get("candidate_id") or c.get("id") or c.get("candidate_id")
            fs = c.get("final_score") or c.get("score") or c.get("combined_score")
            rows.append({"candidate_id": cid, "score": round(fs or 0.0, 4)})
        if rows:
            df = pd.DataFrame(rows).set_index("candidate_id")
            st.table(df)
            pick = st.selectbox(
                "Select candidate to view details",
                options=list(range(len(ranked[:k]))),
                format_func=lambda i: f"{ranked[i].get('candidate_id', ranked[i].get('id','N/A'))} — {round((ranked[i].get('final_score') or ranked[i].get('score') or 0.0),4)}",
            )
            chosen = ranked[pick]
            st.markdown("#### Details for selected candidate")
            if chosen.get("score_result"):
                render_score(chosen["score_result"], top_k=top_k)
            elif (
                chosen.get("per_feature")
                or chosen.get("features")
                or chosen.get("final_score")
            ):
                render_score(chosen, top_k=top_k)
            elif chosen.get("candidate"):
                st.json(chosen["candidate"])
            else:
                st.json(chosen)
        return

    final_score = (
        score_obj.get("final_score")
        or score_obj.get("score")
        or score_obj.get("combined_score")
    )
    if final_score is None:
        st.subheader("Score response (raw)")
        st.json(score_obj)
        return

    st.subheader(f"Final score: {final_score:.4f}")
    per = score_obj.get("per_feature") or score_obj.get("features") or {}
    if not isinstance(per, dict):
        try:
            per = {k: v for k, v in (per or [])}
        except Exception:
            per = {}

    rows = []
    for feat, info in per.items():
        rows.append(
            {
                "feature": feat,
                "weight": float(info.get("final_weight", info.get("weight", 0.0))),
                "sim": float(info.get("semantic_similarity", 0.0)),
                "contribution": float(info.get("contribution", 0.0)),
            }
        )

    if not rows:
        st.info("No per-feature information available in the score object.")
        return

    k = None
    if top_k:
        try:
            k = int(top_k)
        except Exception:
            k = None
    top_k_features = k or score_obj.get("top_k") or None

    rows_sorted = sorted(rows, key=lambda r: r["contribution"], reverse=True)
    rows_display = rows_sorted[:top_k_features] if top_k_features else rows_sorted

    df = pd.DataFrame(rows_display).set_index("feature")
    st.table(df)
    st.bar_chart(df["contribution"])

    for feat_row in rows_display:
        feat = feat_row["feature"]
        info = per.get(feat, {})
        with st.expander(
            f"{feat} — sim {info.get('semantic_similarity',0.0):.3f} — contrib {info.get('contribution',0.0):.3f}"
        ):
            ev = info.get("evidence") or info.get("examples") or []
            if not ev:
                st.write("_No evidence available for this feature._")
            else:
                for i, e in enumerate(ev[:30], 1):
                    st.write(f"{i}. {e}")
