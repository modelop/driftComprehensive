"""OOTB ModelOp Center model to run KS, ES, JS, KL, and summary drift methods on input data"""

import pandas

import modelop.monitors.drift as drift
import modelop.schema.infer as infer
import modelop.utils as utils

logger = utils.configure_logger()

JOB = {}

# modelop.init
def init(job_json: dict) -> None:
    """A function to receive the job JSON and validate schema fail-fast.

    Args:
        job_json (dict): job JSON
    """

    # Extract job JSON
    global JOB
    global numerical_columns, categorical_columns
    JOB = job_json
    infer.validate_schema(job_json) 
    input_schema_definition = infer.extract_input_schema(job_json)
    monitoring_parameters = infer.set_monitoring_parameters(
        schema_json=input_schema_definition, check_schema=True
    )
    numerical_columns = monitoring_parameters["numerical_columns"]
    categorical_columns = monitoring_parameters["categorical_columns"]

def count_categorical_values(df_baseline: pandas.DataFrame, df_sample: pandas.DataFrame) -> dict:
    """A function to compute unique value counts for categorical fields.

    Args:
        df_baseline (pandas.DataFrame): Baseline dataset containing model inputs
        df_sample (pandas.DataFrame): Sample (prod) dataset containing model inputs

    Returns:
        dict: Value counts in graphable form (horizontal bar chart)
    """
    
    baseline_counts = {}
    sample_counts = {}

    # value counts for baseline and sample
    for col in categorical_columns:
        values = df_baseline[col].value_counts()
        for i in values.iteritems():
            count_field = col + '_' + i[0] + '_' + 'count'
            baseline_counts[count_field] = i[1]
        values = df_sample[col].value_counts()
        for i in values.iteritems():
            count_field = col + '_' + i[0] + '_' + 'count'
            sample_counts[count_field] = i[1]
    
    # get union of baseline_counts and sample_counts
    all_count_fields = list(set(baseline_counts.keys()).union(set(sample_counts.keys())))
    all_count_fields.sort()

    # get into correct output form
    data1 = []
    data2 = []
    for i in all_count_fields:
        if baseline_counts.get(i, False):
            data1.append(baseline_counts[i])
        else:
            data1.append(0)
        if sample_counts.get(i, False):
            data2.append(sample_counts[i])
        else:
            data2.append(0)

    output = {}
    output["Categoricals_Unique_Value_Counts"] = {
        "title": "Count for each unique categorical",
        "x_axis_label": "Count",
        "y_axis_label": "Categorical",
        "rotated": True,
        "data": {
            "data1": data1,
            "data2": data2
        },
        "categories": all_count_fields
    }

    return output

# modelop.metrics
def metrics(df_baseline: pandas.DataFrame, df_sample: pandas.DataFrame) -> dict:
    """A function to compute data drift metrics given baseline and sample (prod) datasets

    Args:
        df_baseline (pandas.DataFrame): Baseline dataset containing model inputs
        df_sample (pandas.DataFrame): Sample (prod) dataset containing model inputs

    Returns:
        dict: Data drift metrics (ES, JS, KL, KS, Summary)
    """

    # Initialize DriftDetector
    drift_detector = drift.DriftDetector(
        df_baseline=df_baseline, df_sample=df_sample, job_json=JOB
    )

    # Compute drift metrics
    # Epps-Singleton p-values
    es_drift_metrics = drift_detector.calculate_drift(
        pre_defined_test="Epps-Singleton", flattening_suffix="_es_pvalue"
    )

    # Jensen-Shannon distance
    js_drift_metrics = drift_detector.calculate_drift(
        pre_defined_test="Jensen-Shannon", flattening_suffix="_js_distance"
    )

    # Kullback-Leibler divergence
    kl_drift_metrics = drift_detector.calculate_drift(
        pre_defined_test="Kullback-Leibler",
        flattening_suffix="_kl_divergence",
    )

    # Kolmogorov-Smirnov p-values
    ks_drift_metrics = drift_detector.calculate_drift(
        pre_defined_test="Kolmogorov-Smirnov", flattening_suffix="_ks_pvalue"
    )

    # Pandas summary
    summary_drift_metrics = drift_detector.calculate_drift(pre_defined_test="Summary")

    # Value counts
    value_counts = count_categorical_values(df_baseline, df_sample)

    result = utils.merge(
        es_drift_metrics,
        js_drift_metrics,
        kl_drift_metrics,
        ks_drift_metrics,
        summary_drift_metrics,
        value_counts
    )

    yield result
