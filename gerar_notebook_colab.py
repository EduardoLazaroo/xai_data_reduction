"""
Converte os arquivos Markdown didáticos em notebooks Jupyter/Google Colab (.ipynb)
prontos para importação direta (Upload) no Google Colab.
"""
import json
import os
import re

BASE_DIR = r"c:\Users\eduar\projetos\xai_data_reduction"


def md_to_ipynb(md_path: str, ipynb_path: str) -> int:
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Limpa tags de alerta do GitHub (> [!NOTE], > [!TIP], > [!IMPORTANT], etc.) para renderizar limpo no Colab
    content = re.sub(r"^>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*\n", "", content, flags=re.MULTILINE)

    # Divide o conteúdo entre blocos de código ```python ... ``` e blocos de texto Markdown
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    cells = []
    last_end = 0

    for idx, match in enumerate(pattern.finditer(content)):
        md_chunk = content[last_end:match.start()].strip()
        if md_chunk:
            # Podemos subdividir blocos grandes de markdown em "---" para ficar ainda mais organizado no Colab
            sub_chunks = [c.strip() for c in re.split(r"\n---\n", md_chunk) if c.strip()]
            for sc in sub_chunks:
                lines = [line + "\n" for line in sc.splitlines()]
                if lines:
                    lines[-1] = lines[-1].rstrip("\n")
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {"id": f"md_{len(cells)}"},
                    "source": lines
                })

        code_chunk = match.group(1).strip("\n")
        code_lines = [line + "\n" for line in code_chunk.splitlines()]
        if code_lines:
            code_lines[-1] = code_lines[-1].rstrip("\n")
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": f"code_{len(cells)}"},
            "outputs": [],
            "source": code_lines
        })
        last_end = match.end()

    # Restante final de Markdown
    tail_md = content[last_end:].strip()
    if tail_md:
        sub_chunks = [c.strip() for c in re.split(r"\n---\n", tail_md) if c.strip()]
        for sc in sub_chunks:
            lines = [line + "\n" for line in sc.splitlines()]
            if lines:
                lines[-1] = lines[-1].rstrip("\n")
            cells.append({
                "cell_type": "markdown",
                "metadata": {"id": f"md_{len(cells)}"},
                "source": lines
            })

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "name": os.path.basename(ipynb_path)
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "cells": cells
    }

    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)

    return len(cells)


if __name__ == "__main__":
    # 1. Masterclass principal em notebooks_colab/
    master_md = os.path.join(BASE_DIR, "notebooks_colab", "XAI_Data_Reduction_Masterclass_Colab.md")
    master_ipynb = os.path.join(BASE_DIR, "notebooks_colab", "XAI_Data_Reduction_Masterclass_Colab.ipynb")
    n_cells = md_to_ipynb(master_md, master_ipynb)
    print(f"[+] Gerado: {master_ipynb} ({n_cells} células)")

    # 2. Aulas individuais em aulas_colab/
    aulas_dir = os.path.join(BASE_DIR, "aulas_colab")
    for fname in sorted(os.listdir(aulas_dir)):
        if fname.startswith("Aula_") and fname.endswith(".md"):
            md_p = os.path.join(aulas_dir, fname)
            ipynb_p = os.path.join(aulas_dir, fname.replace(".md", ".ipynb"))
            nc = md_to_ipynb(md_p, ipynb_p)
            print(f"[+] Gerado: {ipynb_p} ({nc} células)")

