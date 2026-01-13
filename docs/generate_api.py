"""
Génération automatique de la documentation API.

- Génère les fichiers pour MkDocs
- Les écrit aussi physiquement dans docs/ pour debug
"""

from pathlib import Path
import pkgutil
import mkdocs_gen_files

import soilfauna

DOCS_DIR = Path("docs")
API_DIR = DOCS_DIR / "api"

nav = mkdocs_gen_files.Nav()

def iter_modules(package):
    for module in pkgutil.walk_packages(
        package.__path__,
        package.__name__ + ".",
    ):
        yield module.name

for module_name in iter_modules(soilfauna):
    if module_name.endswith("__main__"):
        continue

    rel_path = Path(module_name.replace(".", "/") + ".md")
    doc_path = API_DIR / rel_path

    content = (
        f"# `{module_name}`\n\n"
        f"::: {module_name}\n"
    )

    # 1️⃣ Écriture réelle sur disque (debug)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(content, encoding="utf-8")

    # 2️⃣ Écriture pour MkDocs (virtuel)
    with mkdocs_gen_files.open(f"api/{rel_path}", "w") as fd:
        fd.write(content)

    # Navigation
    nav_key = tuple(module_name.split(".")[1:])
    nav[nav_key] = f"api/{rel_path.as_posix()}"

# SUMMARY.md
summary_path = API_DIR / "SUMMARY.md"
summary_content = nav.build_literate_nav()

summary_path.write_text("".join(summary_content), encoding="utf-8")

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(summary_content)