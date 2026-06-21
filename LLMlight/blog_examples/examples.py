# Import library
# Import library
import bnlearn as bn
from distfit import distfit
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = bn.import_example('predictive_maintenance')

# print dataframe
# +-------+------------+------+------------------+----+-----+-----+-----+-----+
# |  UDI | Product ID  | Type | Air temperature  | .. | HDF | PWF | OSF | RNF |
# +-------+------------+------+------------------+----+-----+-----+-----+-----+
# |    1 | M14860      |   M  | 298.1            | .. |   0 |   0 |   0 |   0 |
# |    2 | L47181      |   L  | 298.2            | .. |   0 |   0 |   0 |   0 |
# |    3 | L47182      |   L  | 298.1            | .. |   0 |   0 |   0 |   0 |
# |    4 | L47183      |   L  | 298.2            | .. |   0 |   0 |   0 |   0 |
# |    5 | L47184      |   L  | 298.2            | .. |   0 |   0 |   0 |   0 |
# | ...  | ...         | ...  | ...              | .. | ... | ... | ... | ... |
# | 9996 | M24855      |   M  | 298.8            | .. |   0 |   0 |   0 |   0 |
# | 9997 | H39410      |   H  | 298.9            | .. |   0 |   0 |   0 |   0 |
# | 9998 | M24857      |   M  | 299.0            | .. |   0 |   0 |   0 |   0 |
# | 9999 | H39412      |   H  | 299.0            | .. |   0 |   0 |   0 |   0 |
# |10000 | M24859      |   M  | 299.0            | .. |   0 |   0 |   0 |   0 |
# +-------+-------------+------+------------------+----+-----+-----+-----+-----+
# [10000 rows x 14 columns]

# Remove IDs from Dataframe
del df['UDI']
del df['Product ID']



# Discretize the following columns
colnames = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
colors = ['#87CEEB', '#FFA500', '#800080', '#FF4500', '#A9A9A9']

# Apply distribution fitting to each variable
for colname, color in zip(colnames, colors):
    # Initialize and set 95% confidence interval
    if colname=='Tool wear [min]' or colname=='Process temperature [K]':
        # Set model parameters to determine the medium-high ranges
        dist = distfit(alpha=0.05, bound='up', stats='RSS')
        labels = ['medium', 'high']
    else:
        # Set model parameters to determine the low-medium-high ranges
        dist = distfit(alpha=0.05, stats='RSS')
        labels = ['low', 'medium', 'high']

    # Distribution fitting
    dist.fit_transform(df[colname])

    # Plot
    dist.plot(title=colname, bar_properties={'color': color})
    plt.show()

    # Define bins based on distribution
    bins = [df[colname].min(), dist.model['CII_min_alpha'], dist.model['CII_max_alpha'], df[colname].max()]
    # Remove None
    bins = [x for x in bins if x is not None]

    # Discretize using the defined bins and add to dataframe
    df[colname + '_category'] = pd.cut(df[colname], bins=bins, labels=labels, include_lowest=True)
    # Delete the original column
    del df[colname]

# Structure learning
model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
# [bnlearn] >Warning: Computing DAG with 12 nodes can take a very long time!
# [bnlearn] >Computing best DAG using [hc]
# [bnlearn] >Set scoring type at [bds]
# [bnlearn] >Compute structure scores for model comparison (higher is better).

print(model['structure_scores'])
# {'k2': -23261.534992034045,
# 'bic': -23296.9910477033,
# 'bdeu': -23325.348497769708,
# 'bds': -23397.741317668322}

# Compute edge weights using ChiSquare independence test.
model = bn.independence_test(model, df, test='chi_square', prune=True)

# Plot the best DAG
bn.plot(model, edge_labels='pvalue', params_static={'maxscale': 4, 'figsize': (15, 15), 'font_size': 14, 'arrowsize': 10})

dotgraph = bn.plot_graphviz(model, edge_labels='pvalue')
dotgraph

# Store to pdf
dotgraph.view(filename='bnlearn_predictive_maintanance')


# %%
from LLMlight import LLMlight

# Initialize LLMlight client
client = LLMlight(verbose='info', endpoint="http://localhost:1234/v1/chat/completions")

# Get a list of available models
models = client.get_available_models()
modelinfo = client.get_model_info(model=models[0])

# Print available models
print(modelinfo)
print(models)

# %%
from LLMlight import LLMlight

models = ['nvidia/nemotron-3-nano-omni', 'vibethinker-3b', 'google/gemma-4-26b-a4b-qat', 'qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive', 'openai/gpt-oss-20b', 'unsloth/qwen3-coder-30b-a3b-instruct']

client = LLMlight(verbose='info', endpoint="http://localhost:1234/v1/chat/completions", model=models[0])

query = 'hi'
instructions = ''
system=''
context = ''
response_format=''
temperature=0.2

out = client.prompt(query, instructions=instructions, system=system, context=context, response_format=response_format, temperature=temperature, return_type='max')
out['usage']['total_tokens']

# %% CONTEXT OPTION 1

categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
numeric_cols = df.select_dtypes(include=["number"]).columns

df_numeric = df[numeric_cols]

categorical_summary = "\n\n".join(
    f"{col}:\n{df[col].value_counts(dropna=False).to_string()}"
    for col in categorical_cols
)

target = "Machine failure"

target_summary = "\n\n".join(
    f"{col} by {target}:\n"
    f"{pd.crosstab(df[col], df[target], normalize='index').round(3).to_string()}"
    for col in df.columns
    if col != target and df[col].nunique() <= 20
)



context = f"""
DATASET OVERVIEW
Rows: {df.shape[0]}
Columns: {df.shape[1]}

VARIABLES

1. Type
   Product quality/type category.
   Possible values: L, M, H.
   This is an upstream design or product-related variable.

2. Air temperature [K]
   Ambient air temperature around the machine.

3. Process temperature [K]
   Operating process temperature of the machine.

4. Rotational speed [rpm]
   Machine spindle or motor rotational speed.

5. Torque [Nm]
   Mechanical torque/load on the machine.

6. Tool wear [min]
   Accumulated tool usage or wear time.

7. Machine failure
   Binary outcome variable indicating whether a machine failure occurred.

8. TWF
   Tool wear failure indicator.

9. HDF
   Heat dissipation failure indicator.

10. PWF
   Power failure indicator.

11. OSF
   Overstrain failure indicator.

12. RNF
   Random failure indicator.

Additional pre-processed categorical variables may be present, for example:

- Air temperature [K]_category
- Process temperature [K]_category
- Rotational speed [rpm]_category
- Torque [Nm]_category
- Tool wear [min]_category
IMPORTANT DATA RULE
Variables ending with "_category" are derived from numeric variables.
They should not be treated as independent causes.

COLUMN TYPES
{df.dtypes.to_string()}

MISSING VALUES
{df.isna().sum().to_string()}

UNIQUE VALUES
{df.nunique().to_string()}

VALUE COUNTS FOR CATEGORICAL VARIABLES
{categorical_summary}

NUMERIC SUMMARY
{df.describe().to_string()}

CORRELATION MATRIX
{df_numeric.corr().round(3).to_string()}

TARGET RELATIONSHIPS
{target_summary}

SMALL REPRESENTATIVE SAMPLE
{df.sample(30, random_state=42).to_string()}
"""

# %% CONTEXT OPTION 2
from sklearn.feature_selection import mutual_info_classif
import pandas as pd

evidence = {
    "shape": df.shape,
    "columns": list(df.columns),
    "dtypes": df.dtypes.astype(str).to_dict(),
    "missing": df.isna().sum().to_dict(),
    "unique": df.nunique().to_dict(),
    "value_counts": {
        col: df[col].value_counts(dropna=False).head(20).to_dict()
        for col in df.columns
        if df[col].nunique() <= 30
    },
    "numeric_summary": df.describe().round(3).to_dict(),
    "correlation": df.select_dtypes("number").corr().round(3).to_dict(),
}


X = pd.get_dummies(df.drop(columns=["Machine failure"]), drop_first=False)
y = df["Machine failure"]

mi = mutual_info_classif(X, y, discrete_features="auto", random_state=42)

mi_summary = (
    pd.DataFrame({"variable": X.columns, "mutual_information": mi})
    .sort_values("mutual_information", ascending=False)
    .head(30)
)

context = f"""
You are given evidence computed from the full dataset, not a sample.

The original dataset has {df.shape[0]} rows and {df.shape[1]} columns.

The following summaries were computed using all rows.

DATASET EVIDENCE
{evidence}

MUTUAL INFORMATION WITH MACHINE FAILURE
{mi_summary.to_string(index=False)}

Important:
- Use this evidence to infer a causal DAG.
- Do not treat correlation as causation.
- Use domain reasoning to orient edges.
- Variables ending with '_category' are derived variables.
- Machine failure is the outcome.
"""

# %%

query = """
Determine the most plausible causal structure for the predictive maintenance dataset.

Your task is to infer a Directed Acyclic Graph (DAG) that explains the causal relations between the variables.
Focus on causal mechanisms, not only correlations.
"""

instructions = """
You are given a pre-processed predictive maintenance dataset with 10,000 rows and 12 variables.

Infer the causal structure between the variables.

Requirements:
1. Return a Directed Acyclic Graph (DAG).
2. Use domain reasoning from predictive maintenance and machine behavior.
3. Distinguish causes, effects, mediators, and target variables.
4. Do not create cycles.
5. Do not infer causal relations only because variables are correlated.
6. Treat 'Machine failure' as the main outcome variable.
7. Explain each proposed causal edge briefly.
8. Mention uncertain edges separately.
9. Do not use hidden variables unless strictly necessary.
10. Prefer a compact, interpretable causal graph.

Important:
The dataset is pre-processed and contains discretized category variables. These category variables represent binned versions of the original continuous measurements.
"""

system = """
You are an expert in causal discovery, Bayesian networks, predictive maintenance, and industrial machine learning.

You reason carefully about causal direction, confounding, mediation, and domain constraints.
You do not hallucinate causal relations.
You only propose edges that are plausible given the variables and the predictive maintenance context.
"""

response_format = """
Return the answer as valid JSON with the following structure:

{
  "nodes": [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "Machine failure"
  ],
  "edges": [
    {
      "source": "variable_name",
      "target": "variable_name",
      "reason": "brief causal explanation",
      "confidence": "high | medium | low"
    }
  ],
  "excluded_variables": [
    {
      "variable": "variable_name",
      "reason": "why this variable was excluded or treated as derived"
    }
  ],
  "uncertain_edges": [
    {
      "source": "variable_name",
      "target": "variable_name",
      "reason": "why the edge is uncertain"
    }
  ],
  "causal_summary": "short explanation of the overall causal structure"
}
"""

# %%
temperature = 0.2

out = client.prompt(
    query,
    instructions=instructions,
    system=system,
    context=context,
    response_format=response_format,
    temperature=temperature,
    return_type='raw')

print(f"Tokens used: {out['usage']['total_tokens']}")
