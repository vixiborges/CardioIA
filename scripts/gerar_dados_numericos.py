"""
CardioIA - Fase 1: Batimentos de Dados
Gerador de dataset simulado de pacientes cardiológicos (dados de IoT/sensores).

Este script gera um dataset SINTÉTICO (não contém dados reais de pacientes),
mas segue distribuições clinicamente plausíveis, baseadas em faixas de
referência amplamente documentadas na literatura médica (ex.: American Heart
Association, Sociedade Brasileira de Cardiologia), para servir de base
verossímil para as próximas fases do projeto (ML, dashboards, alertas).

Autor: Gustavo (FIAP - Engenharia de Software)
"""

import numpy as np
import pandas as pd

# Seed fixa para reprodutibilidade (importante para governança/documentação)
SEED = 42
rng = np.random.default_rng(SEED)

N = 250  # número de pacientes simulados (min. exigido: 100)


def gerar_dataset(n: int) -> pd.DataFrame:
    # --- Variáveis demográficas ---
    idade = rng.integers(18, 90, size=n)
    sexo = rng.choice(["M", "F"], size=n, p=[0.49, 0.51])

    # --- Fatores de risco comportamentais ---
    tabagismo = rng.choice(["Sim", "Não", "Ex-fumante"], size=n, p=[0.18, 0.62, 0.20])
    atividade_fisica = rng.choice(
        ["Sedentário", "Leve", "Moderada", "Intensa"], size=n, p=[0.35, 0.30, 0.25, 0.10]
    )

    # IMC correlacionado levemente com sedentarismo
    imc_base = rng.normal(26, 4.5, size=n)
    imc_ajuste = np.where(atividade_fisica == "Sedentário", 1.5, 0)
    imc = np.clip(imc_base + imc_ajuste, 16, 45).round(1)

    # --- Pressão arterial (correlacionada com idade e IMC) ---
    pas_base = 100 + (idade * 0.35) + (imc - 25) * 0.8 + rng.normal(0, 8, size=n)
    pad_base = 65 + (idade * 0.15) + (imc - 25) * 0.4 + rng.normal(0, 6, size=n)
    pressao_sistolica = np.clip(pas_base, 90, 210).round(0).astype(int)
    pressao_diastolica = np.clip(pad_base, 55, 130).round(0).astype(int)

    # --- Colesterol (correlacionado com idade, IMC e tabagismo) ---
    fumante_flag = np.where(tabagismo == "Sim", 15, 0)
    colesterol_total = np.clip(
        150 + (idade * 0.6) + (imc - 25) * 1.2 + fumante_flag + rng.normal(0, 20, size=n),
        120, 340
    ).round(0).astype(int)
    colesterol_hdl = np.clip(rng.normal(50, 12, size=n) - fumante_flag * 0.3, 20, 90).round(0).astype(int)
    colesterol_ldl = np.clip(colesterol_total - colesterol_hdl - rng.integers(10, 40, size=n), 40, 250).astype(int)

    # --- Glicemia ---
    glicemia_jejum = np.clip(85 + (imc - 25) * 1.1 + rng.normal(0, 15, size=n), 65, 260).round(0).astype(int)

    # --- Frequência cardíaca (bpm) ---
    freq_cardiaca = np.clip(rng.normal(75, 11, size=n) + (atividade_fisica == "Sedentário") * 4, 45, 140).round(0).astype(int)

    # --- Histórico familiar ---
    historico_familiar = rng.choice(["Sim", "Não"], size=n, p=[0.32, 0.68])

    # --- Sintomas relatados (múltiplos possíveis, simplificado em categoria dominante) ---
    sintomas = rng.choice(
        ["Nenhum", "Dor no peito", "Falta de ar", "Palpitações", "Tontura", "Fadiga"],
        size=n, p=[0.40, 0.15, 0.15, 0.12, 0.08, 0.10]
    )

    # --- Score de risco latente (para gerar o diagnóstico de forma coerente) ---
    score = (
        (idade > 55).astype(int) * 1.5
        + (pressao_sistolica > 140).astype(int) * 1.5
        + (colesterol_total > 240).astype(int) * 1.2
        + (colesterol_ldl > 160).astype(int) * 1.0
        + (imc > 30).astype(int) * 0.8
        + (tabagismo == "Sim").astype(int) * 1.3
        + (historico_familiar == "Sim").astype(int) * 1.0
        + (atividade_fisica == "Sedentário").astype(int) * 0.7
        + (glicemia_jejum > 126).astype(int) * 0.9
        + (sintomas != "Nenhum").astype(int) * 0.6
        + rng.normal(0, 1.0, size=n)
    )
    limiar = np.quantile(score, 0.65)
    diagnostico_doenca_cardiaca = np.where(score > limiar, "Positivo", "Negativo")

    df = pd.DataFrame({
        "id_paciente": [f"PAC{str(i+1).zfill(4)}" for i in range(n)],
        "idade": idade,
        "sexo": sexo,
        "pressao_sistolica_mmHg": pressao_sistolica,
        "pressao_diastolica_mmHg": pressao_diastolica,
        "colesterol_total_mgdl": colesterol_total,
        "colesterol_hdl_mgdl": colesterol_hdl,
        "colesterol_ldl_mgdl": colesterol_ldl,
        "glicemia_jejum_mgdl": glicemia_jejum,
        "frequencia_cardiaca_bpm": freq_cardiaca,
        "imc": imc,
        "tabagismo": tabagismo,
        "atividade_fisica": atividade_fisica,
        "historico_familiar_cardiaco": historico_familiar,
        "sintomas_relatados": sintomas,
        "diagnostico_doenca_cardiaca": diagnostico_doenca_cardiaca,
    })
    return df


if __name__ == "__main__":
    df = gerar_dataset(N)
    saida = "../data/pacientes_cardiacos_simulado.csv"
    df.to_csv(saida, index=False, encoding="utf-8")
    print(f"Dataset gerado com {len(df)} linhas e {len(df.columns)} colunas.")
    print(f"Salvo em: {saida}")
    print("\nPrévia:")
    print(df.head())
    print("\nDistribuição do diagnóstico:")
    print(df['diagnostico_doenca_cardiaca'].value_counts())
