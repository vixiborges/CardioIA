# CardioIA — Fase 1: Batimentos de Dados

> Projeto acadêmico FIAP (PBL) — CardioIA: A Nova Era da Cardiologia Inteligente
> Fase 1 de 7 — Levantamento e organização de dados fundamentais (Governança de Dados)

## 🎯 Objetivo da fase

Levantar, organizar e documentar três tipos de dados que servirão de base para
os módulos inteligentes das próximas fases do CardioIA (Machine Learning, NLP
e Visão Computacional): **dados numéricos**, **dados textuais** e
**imagens médicas**, com atenção explícita a Governança de Dados e a
possíveis vieses.

**Autor:** Gustavo — [RM: 567477]
**GitHub:** https://github.com/vixiborges/CardioIA

---

## 📁 Estrutura do repositório

```
cardioia-fase1/
├── README.md                          <- este arquivo
├── data/
│   └── pacientes_cardiacos_simulado.csv
├── scripts/
│   └── gerar_dados_numericos.py       <- script que gera o CSV
├── docs/
│   ├── warren_1809_doencas_cardiacas.txt   
│   └── manejo_dcv_mulheres_2023.txt        
└── assets/
    └── (pasta de imagens, ou link para Drive/OneDrive)
```

---

## Parte 1 — Dados Numéricos (IoT)

**Arquivo:** `data/pacientes_cardiacos_simulado.csv`
**Link para os dados completos (Drive/OneDrive):** [preencher após upload]

### Origem dos dados
Os dados são **100% simulados**, gerados pelo script `scripts/gerar_dados_numericos.py`
com `numpy`/`pandas`, seed fixa (42) para reprodutibilidade. Optei por dados
sintéticos em vez de uma base real de pacientes por dois motivos: (1) evitar
qualquer risco de exposição de dados sensíveis de saúde (LGPD/HIPAA) já nesta
fase inicial, e (2) ter controle total sobre as distribuições estatísticas,
permitindo documentar exatamente como cada variável foi construída — um
exercício direto de Governança de Dados.

As distribuições e correlações (ex.: pressão arterial subindo com idade e
IMC; colesterol subindo com tabagismo) foram calibradas para serem
clinicamente plausíveis, com base em faixas de referência conhecidas na
literatura de cardiologia, mas **não representam pacientes reais** e não
devem ser usadas para qualquer finalidade clínica.

### Variáveis do dataset (250 registros, 16 colunas)

| Variável | Descrição | Relevância clínica |
|---|---|---|
| `idade` | Idade do paciente (18–89) | Fator de risco não-modificável mais forte para DCV |
| `sexo` | M/F | Perfis de risco cardiovascular diferem entre sexos |
| `pressao_sistolica_mmHg` / `pressao_diastolica_mmHg` | Pressão arterial | Hipertensão é o principal fator de risco modificável |
| `colesterol_total_mgdl`, `colesterol_hdl_mgdl`, `colesterol_ldl_mgdl` | Perfil lipídico | LDL alto e HDL baixo estão associados a aterosclerose |
| `glicemia_jejum_mgdl` | Glicemia em jejum | Diabetes é comorbidade frequente em doença cardíaca |
| `frequencia_cardiaca_bpm` | Frequência cardíaca de repouso | Alterações indicam arritmias ou descondicionamento |
| `imc` | Índice de Massa Corporal | Obesidade correlaciona com hipertensão e diabetes |
| `tabagismo` | Sim/Não/Ex-fumante | Fator de risco modificável de maior impacto após hipertensão |
| `atividade_fisica` | Nível de sedentarismo | Fator protetor bem documentado |
| `historico_familiar_cardiaco` | Sim/Não | Componente genético do risco cardiovascular |
| `sintomas_relatados` | Sintoma predominante | Insumo direto para triagem (Fase futura de agentes inteligentes) |
| `diagnostico_doenca_cardiaca` | Positivo/Negativo | Variável-alvo (label) para os modelos de ML das próximas fases |

**Variáveis mais relevantes clinicamente:** `pressao_sistolica_mmHg`,
`colesterol_ldl_mgdl` e `historico_familiar_cardiaco` foram consideradas as
mais importantes, pois concentram os fatores de risco modificável e
não-modificável mais citados nas diretrizes de cardiologia (hipertensão,
dislipidemia e genética), e serão prováveis features de maior peso nos
modelos preditivos das fases seguintes.

### ⚠️ Nota de Governança de Dados e Viés
Por serem sintéticos, esses dados **não carregam o viés social real** de
bases hospitalares (ex.: sub-representação de determinados grupos
socioeconômicos ou étnicos no acesso a diagnóstico). Isso é uma limitação
importante a se documentar: um modelo de ML treinado só com este dataset
não deve ser generalizado para decisões clínicas reais, pois os dados
sintéticos, embora clinicamente plausíveis, não capturam desigualdades de
acesso à saúde presentes em dados populacionais reais. Essa discussão será
retomada nas fases de Machine Learning do projeto.

---

## Parte 2 — Dados Textuais (NLP)

**Local:** `docs/`
**Guia completo de fontes e links de download:** [`docs/fontes_textos.md`](docs/fontes_textos.md)

Dois textos foram selecionados, ambos com licenciamento que permite
redistribuição:

1. **Manejo das Doenças Cardiovasculares em Mulheres** (2023) — artigo
   científico de acesso aberto (CC BY), Arquivos Brasileiros de Cardiologia.
2. **Cases of Organic Diseases of the Heart** (1809), de John Collins Warren —
   obra histórica em domínio público, via Project Gutenberg.

### Como esses textos podem ser explorados por NLP
- **Extração de sintomas e entidades clínicas**: identificar menções a dor
  no peito, palpitações, dispneia, etc., útil para alimentar futuros
  módulos de triagem automatizada por texto livre;
- **Classificação de tópicos**: separar trechos sobre epidemiologia,
  diagnóstico, tratamento e prevenção;
- **Análise de sentimento/tom**: comparar a linguagem médica descritiva do
  século XIX com a linguagem científica atual, evidenciando a evolução do
  discurso médico sobre doenças cardíacas;
- **Nuvem de palavras / TF-IDF**: mapear os termos mais recorrentes em cada
  fonte como primeira exploração exploratória de dados textuais.

Essas análises são relevantes para IA em saúde porque prontuários e
anotações clínicas em texto livre são uma fonte massiva e subutilizada de
informação — algoritmos de NLP podem transformar esse texto não estruturado
em dados estruturados utilizáveis por sistemas de apoio à decisão clínica.

---

## Parte 3 — Dados Visuais (Visão Computacional)

**Local:** `assets/`
**Link para as imagens completas (Drive/OneDrive):** [https://drive.google.com/drive/folders/1c0G793aUDAYeJUm9J2KieNyWy-axbolC?usp=drive_link]

Foram selecionadas fontes públicas de imagens de ECG, todas com licença compatível com uso
acadêmico (CC BY 4.0, MIT, ou domínio público via NIH/PhysioNet).

### Como essas imagens podem ser exploradas por Visão Computacional
- **Detecção de bordas e segmentação**: isolar a silhueta cardíaca ou o
  traçado do ECG para análise automatizada (ex.: filtros de Sobel, U-Net);
- **Reconhecimento de padrões/anomalias**: treinar classificadores (CNNs)
  para distinguir traçados normais de arritmias, ou identificar estenoses em
  angiogramas;
- **Métricas quantitativas automatizadas**: por exemplo, o índice
  cardiotorácico (CTR) em raio-X de tórax, hoje calculado manualmente por
  radiologistas, é um bom candidato a automação;
- **Relevância para IA em saúde**: a análise automatizada de imagens médicas
  permite triagem mais rápida e escalável, especialmente relevante em
  regiões com poucos especialistas em cardiologia — um dos pilares do
  CardioIA nas fases futuras de Visão Computacional.

---

## Governança de Dados — resumo geral

| Aspecto | Como foi tratado |
|---|---|
| Dados sensíveis reais | Evitados nesta fase — dados numéricos são 100% sintéticos |
| Licenciamento | Todos os textos e imagens usados têm licença que permite uso acadêmico/redistribuição (CC BY, MIT, domínio público) |
| Viés | Documentado explicitamente que os dados sintéticos não capturam desigualdades reais de acesso à saúde |
| Reprodutibilidade | Script de geração de dados com seed fixa, versionado no repositório |
| Atribuição | Toda fonte externa (texto e imagem) é referenciada com link e licença |

## Como reproduzir

```bash
pip install pandas numpy
cd scripts
python3 gerar_dados_numericos.py
```

## Próximas fases

Este dataset e materiais servirão de base para as fases seguintes do
CardioIA: modelos de Machine Learning para triagem, NLP para análise de
prontuários, Visão Computacional para leitura automatizada de exames, IoT
para monitoramento contínuo e agentes inteligentes para assistência remota.
