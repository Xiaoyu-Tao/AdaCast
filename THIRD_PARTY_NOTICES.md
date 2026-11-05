# Third-party dependencies

Application code is maintained in this project. Runtime schema validation uses Zod (MIT). Bun provides the runtime, SQLite and compiler. TypeScript and Bun/Node type definitions are development dependencies; see the installed packages' licenses.

Optional Python model integrations use PyTorch, statsmodels and Chronos through their public interfaces. Their code, model weights, terms and licenses remain with their respective maintainers. The dependency lock files identify exact resolved packages.

The numerical engine, forecasting domain rules and role prompts include MIT-licensed implementations incorporated from the AdaCast project (Copyright 2026 Anonymous Authors). Their license text is retained at `python/src/adacast_lite_ml/engine/LICENSE` and `src/domain/LICENSE`. These are independently owned source copies in this distribution; no original package is loaded at runtime. The compiled release also includes `ADACAST_ENGINE_LICENSE.txt`.

Runtime Markdown frontmatter parsing uses gray-matter (MIT). Scientific dependencies include NumPy, pandas, SciPy, scikit-learn, statsmodels, matplotlib, einops and tqdm; their respective licenses apply.
