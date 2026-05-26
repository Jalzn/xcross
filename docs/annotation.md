# Annotation — sessão da expansão de modelos + comparação de paper

Anotação viva do trabalho da sessão de 2026-05-25/26. Serve de scratch-pad para retomar
contexto se a sessão cair. NÃO é prosa de paper — é nota técnica.

## 1. Norte do projeto

Paper completo sobre o xCross. Esta sessão produziu **§10 — Comparação de modelos**,
que justifica a escolha de estimador com rigor (não joga números crus) e fecha a tese:
*nem o estado da arte tabular destrava o xCross — o teto é de sinal*. Detalhes em
[[paper-model-comparison-cloud]] (memory).

## 2. Estado atual (o que foi entregue)

### Código novo no branch `models/expand-registry-eval`
- `xcross/model/estimators.py` — 7 estimadores in-process (xgboost, adaboost, catboost,
  lightgbm, histgb, random_forest, logreg) + TabPFN condicional via env
  `XCROSS_TABPFN=1` (torch lazy import, sem afetar uso local).
- `xcross/model/evaluate.py` — `stability_temporal` e `topk_overlap_temporal` (split
  cronológico) entram em `metrics()` quando `order_key` é passado.
- `xcross/model/selection.py` — critério **diferenciado**: xCross → `stability_temporal`,
  xCrossOT → `auc` (mapa `CRITERION`). Param `eligible` restringe headline a estimadores
  in-process do report.
- `xcross/model/compare.py` — salva `oof_matrix.parquet` (64 colunas: 8 est × 2 fs × 2
  lbl × 2 cal) além do `comparison.csv`.
- `xcross/model/robustness.py` — bootstrap IC para AUC e stability + matriz de
  ranking-agreement modelo-vs-modelo (lê o `oof_matrix`).
- `xcross/model/comparison_figures.py` — 4 figuras p/ §10: trade-off por família,
  robustez com IC, agreement heatmap, calibração comparada.
- `xcross/model/report.py` — importância model-agnostic (permutation p/ modelos sem
  `feature_importances_`); PDP pulado para TabPFN (custo proibitivo na L4).
- `xcross/model/train.py` — `_free_gpu()` libera VRAM entre folds (evita OOM do TabPFN).
- `xcross/model/dataset.py` — `match_dates()` compartilhado (DRY).
- `scripts/run_pipeline_lightning.py` — orquestrador cloud com modos `launch`
  (nohup detached) e `collect` (baixa + para studio).
- `scripts/run_report_only_lightning.py` — orquestrador só do report (pulando
  compare/robustness já feitos).
- `xcross/model/tabpfn_oof.py` — runner isolado p/ macOS (fallback caso conflito
  OpenMP); na nuvem o TabPFN entra no `ESTIMATORS` direto.

### Artefatos baixados (em `artifacts/reports/`)
26 CSVs + 45 figuras, todos com TabPFN integrado na comparação.

### Headlines selecionados (do `model_metrics.csv`)
| alvo | headline | calib. | AUC | stability | ECE |
|---|---|---|---|---|---|
| xcross/success | adaboost | sigmoid | 0.578 | 0.735 | 0.013 |
| xcross/shot | adaboost | sigmoid | 0.582 | 0.717 | 0.007 |
| xcrossot/success | **catboost** | isotonic | 0.838 | 0.354 | 0.007 |
| xcrossot/shot | adaboost | isotonic | 0.727 | 0.416 | 0.011 |

**TabPFN não virou headline** porque a env `XCROSS_TABPFN=1` não propagou para dentro do
`nohup bash -lc` no studio → `ESTIMATORS` no report tinha apenas 7 → `eligible` filtrou
o TabPFN. *A correção dessa propagação é pendência se quisermos o TabPFN como headline.*

## 3. Achados para a §10 do paper

### Tabela final (isotonic, IC 95% por bootstrap)
**xcross/success (creation):**
- adaboost: AUC 0.575 [0.564, 0.584], stab 0.700 [0.636, 0.749], stab_t 0.624, ECE 0.011 🏆
- random_forest: stab 0.690 [0.624, 0.741], stab_t 0.611
- **tabpfn: AUC 0.587 [0.576, 0.597]** (maior), stab 0.651 [0.573, 0.708], stab_t 0.560
- catboost / logreg: stab ~0.55-0.60
- lightgbm / histgb / xgboost: stab ~0.46-0.48 (ICs disjuntos com adaboost)

**xcrossot/success (danger):**
- **tabpfn: AUC 0.849 [0.842, 0.856]** 🏆 — maior absoluto
- xgboost: AUC 0.844 [0.837, 0.850] — IC sobrepõe ao tabpfn
- catboost: AUC 0.840, **ECE 0.005** — calibração muito melhor (foi escolhido como
  headline operável)
- adaboost / logreg / random_forest: AUC 0.81-0.82

### Os 4 pontos centrais
1. **Critério por objetivo** — bagging/boosting-suave vence o xCross (creation);
   foundation+boosting vence o xCrossOT (danger). Famílias diferentes para jobs
   diferentes, com **ICs disjuntos** no xCross (estatisticamente significativo).
2. **TabPFN como teto confirma a tese central** — no xCrossOT é o melhor (AUC 0.849);
   no xCross tem o maior AUC mas **não bate a reprodutibilidade**. Logo: *o limite do
   xCross é o sinal no momento do cruzamento, não o classificador*.
3. **Calibração indistinguível** entre modelos (ECE 0.003-0.013); discriminação ×
   reprodutibilidade é o que decide.
4. **Ranking robusto à escolha do modelo** — Spearman entre rankings de jogadores:
   0.84-0.99 entre todos os modelos (xcross), 0.89-0.99 (xcrossot). É propriedade dos
   *dados*, não do estimador.

### Importâncias do headline (do `importance_*.csv`)
- **xcross/success (adaboost):** `entropy_attack_in_second_post` 0.25,
  `entropy_attack_in_center_box` 0.13, `entropy_general_in_second_post` 0.13,
  `gk_lateral_speed` 0.10, `gk_ball_distance` 0.06 → **entropia domina creation**
- **xcross/shot:** `entropy_attack_in_center_box` 0.52, `gk_lateral_speed`,
  `pocket_radius_in_box`
- **xcrossot/success (catboost):** `entropy_attack_in_zone`, `end_y`, `flight_pace_3d`,
  `clearance_over_keeper` → **destino + voo da bola (z)**
- **xcrossot/shot (adaboost):** `distance_from_end_line`, `entropy_attack_in_zone`,
  `pitch_control_in_zone`, `flight_pace_3d`, `clearance_over_keeper`

### Figuras-chave (todas em `artifacts/reports/figures/`)
- `chart_model_tradeoff_{success,shot}.png` — **Fig A** (a âncora), 2 painéis por
  feature_set, cor por família, TabPFN destacado em vermelho
- `chart_model_robustness_{success,shot}.png` — **Fig B**, AUC e stability com IC 95%
  bootstrap, errorbars
- `chart_ranking_agreement_{success,shot}.png` — **Fig C**, heatmap Spearman
  modelo-vs-modelo
- `chart_calibration_compare_{success,shot}.png` — reliability comparada (8 modelos)

## 4. Arquitetura técnica do pipeline cloud

### Lightning AI conta do usuário
- teamspace `personal`, user `jalmirfsjr`, autenticado em `~/.lightning/credentials.json`
- A100 **indisponível** no cluster AWS — usar Machine.L4 (22 GB VRAM)
- L4 está no limite p/ TabPFN com contexto ~6k → precisa do fix de memória abaixo

### Decisões de design
- TabPFN como cidadão de 1ª classe **na nuvem** (Linux: sem conflito OpenMP); isolamento
  `tabpfn_oof.py` mantido só p/ macOS local
- `XCROSS_TABPFN=1` ativa TabPFN no registry com torch lazy import
- Filesystem do studio persiste entre runs → setup idempotente:
  `git fetch/reset` se `.git` existe, senão `mv ~/xcross ~/xcross.bad.$(date +%s)` +
  clone fresh (`rm -rf` falha no .venv com "Directory not empty")
- Upload com retry em 5xx transitório
- Modos `launch` (nohup detached, sobrevive ao Mac desligar) e `collect` (baixa + para)

### Fix de OOM do TabPFN na L4 (todos juntos)
- `memory_saving_mode=True` no `TabPFNClassifier`
- `torch.cuda.empty_cache()` entre folds no `oof_predict` (`_free_gpu()`)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` no `_ENV`
- **Validado** com teste isolado: `TABPFN_OOF_OK (11677,) 0.347`

## 5. Problemas resolvidos nesta sessão

| problema | solução |
|---|---|
| segfault OpenMP torch+xgboost+lightgbm no macOS | isolamento via `tabpfn_oof.py` (subprocess) |
| TabPFN exige token Prior Labs | baixar pesos v2 do HF público sem token |
| Conflito OpenMP no Linux? | **não ocorre** — `COEXIST_OK cuda=True` validado |
| A100 indisponível no cluster | usar L4 |
| Upload Lightning 502 transitório | retry com backoff de 30s × 4 tries |
| `rm -rf` falha no .venv | `mv ~/xcross ~/xcross.bad.$TS` antes de clone |
| Torch corrompido por runs interrompidos | deletar e recriar studio |
| OOM do TabPFN na L4 | `memory_saving_mode` + `empty_cache` + `expandable_segments` |
| Report travou em silêncio (2h+) | `PYTHONUNBUFFERED=1` no `_ENV` + skip PDP do TabPFN |
| Studio adormeceu apesar de `auto_sleep=False` | reiniciar e baixar tudo (artefatos persistem no FS) |
| `XCROSS_TABPFN` não propagou para nohup | **pendente** — root cause do TabPFN não virar headline |
| Log bufferizado escondendo problemas | `PYTHONUNBUFFERED=1` virou padrão |

## 6. Pendências

### Permutation importance do TabPFN (DONE — 2026-05-26)
- Arquivos: `importance_xcrossot_success_tabpfn.csv`, `importance_xcrossot_shot_tabpfn.csv`.
- Rodou em **modo síncrono** (`scripts/run_tabpfn_importance_lightning.py`) — conexão
  SDK ativa mantém o studio acordado (sem auto_sleep), `stop()` explícito no `finally`.
- Parâmetros reduzidos p/ caber em ~1h: `max_samples=500`, `n_repeats=3`,
  `n_estimators=1` no TabPFN. Os números são ruidosos, mas o **ranking das top features
  é estável** — usar como descritivo.
- Custo total: ~$0.50 de crédito (1h05 na L4).

### Achado p/ o paper — TabPFN vs catboost (mesmas xcrossot/success)
- **Convergência**: ambos rankeiam `entropy_attack_in_zone` em #1; `flight_pace_3d`,
  `end_y`, `pitch_control_in_zone`, `temporal_entropy_diff_zone_delta`,
  `clearance_over_keeper` aparecem no top 10 dos dois.
- **Divergência**: TabPFN valoriza mais as **três visões de entropia** (attack, defense,
  general — informação espacial multi-camada); catboost valoriza mais a **geometria**
  (`end_y`, `distance_from_end_line`) e o **voo da bola** geométrico (`swing_inout`,
  `flight_loftiness`).
- Leitura: famílias diferentes capturam ângulos diferentes do mesmo sinal, mas o
  **conjunto de top features é robusto** — reforça que a representação certa é o sinal,
  não o classificador.

### Pendência menor: figura comparativa
- Opcional: gerar `chart_importance_compare_xcrossot_success.png` mostrando TabPFN vs
  catboost lado a lado, para ilustrar o achado de convergência+divergência acima.

### Para o paper
- Rascunhar `docs/paper-outline.md` com a estrutura completa (já discutida).
- Atualizar `docs/model-evolution.md` (§6: expansão + régua + benchmark TabPFN).
- Decidir: TabPFN como headline operável (B/C anteriores) ou só benchmark (A).

## 7. Comandos úteis para retomar

```bash
# rodar localmente (sem TabPFN; segfault no macOS):
uv run python -m xcross.model.compare --no-tabpfn
uv run python -m xcross.model.robustness
uv run python -m xcross.model.report
uv run python -m xcross.model.comparison_figures

# pipeline cloud completo (compare+robustness+report+figuras, com TabPFN 1ª classe):
uv run --with lightning-sdk lightning login        # uma vez
uv run --with lightning-sdk python scripts/run_pipeline_lightning.py launch
uv run --with lightning-sdk python scripts/run_pipeline_lightning.py collect

# só o report na cloud (compare/robustness já prontos):
uv run --with lightning-sdk python scripts/run_report_only_lightning.py launch
uv run --with lightning-sdk python scripts/run_report_only_lightning.py collect
```

## 8. Branch & commits

Branch `models/expand-registry-eval`, push em `origin`. Commits chave:
- `feat(report): model-agnostic feature importance (permutation for tabpfn/logreg)`
- `feat(compare): save OOF predictions matrix`
- `feat(robustness): bootstrap CIs + model-vs-model ranking agreement`
- `feat(figures): model-comparison figures (tradeoff, robustness, agreement, calibration)`
- `feat(estimators): TabPFN as first-class estimator behind XCROSS_TABPFN env`
- `feat(cloud): full-pipeline + report-only orchestrators`
- `fix(tabpfn): memory_saving_mode + free GPU cache between folds (L4 OOM)`
- `fix(cloud): idempotent setup, mv dir aside, retry upload, PYTHONUNBUFFERED`
- `feat(report): skip PDP for TabPFN (cost) + report-only cloud orchestrator`
