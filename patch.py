with open('python/gui_app.py', 'r') as f:
    content = f.read()

search = """    logistic_summary = model_tables["logistic_regression_summary"]
    if logistic_summary is not None and not logistic_summary.empty:
        acc = float(logistic_summary.iloc[0]["accuracy_at_0_5"])
        pseudo_r2 = float(logistic_summary.iloc[0]["mcfadden_pseudo_r2"])
        st.markdown(
            f"- Logistic model summary: accuracy={acc:.3f}, McFadden R²={pseudo_r2:.3f}."
        )"""

replace = """    logistic_summary = model_tables["logistic_regression_summary"]
    if logistic_summary is not None and not logistic_summary.empty:
        acc = float(logistic_summary.iloc[0]["accuracy_at_0_5"])
        pseudo_r2 = float(logistic_summary.iloc[0]["mcfadden_pseudo_r2"])
        st.markdown(
            f"- Logistic model summary: accuracy={acc:.3f}, McFadden R²={pseudo_r2:.3f}."
        )

    anova_summary = model_tables.get("anova_concentration_summary")
    if anova_summary is not None and not anova_summary.empty:
        significant_anova = anova_summary[anova_summary["p_value"] < 0.05]
        if not significant_anova.empty:
            for _, row in significant_anova.iterrows():
                st.markdown(
                    f"- ANOVA significant finding for exposure **{row['exposure']}**: "
                    f"Concentration affects **{row['metric']}** "
                    f"(p={row['p_value']:.4g})."
                )"""

new_content = content.replace(search, replace)

with open('python/gui_app.py', 'w') as f:
    f.write(new_content)
