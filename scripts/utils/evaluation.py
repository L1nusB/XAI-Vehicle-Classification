import pandas
import json
import os
from pathlib import Path

from mmcls.apis.test import single_gpu_test
from mmcls.datasets.builder import build_dataloader
from mmcls.models.builder import build_classifier
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv import Config
from mmseg.datasets.builder import build_dataset

from .io import save_json, get_samples, generate_filtered_annfile

def compare_original_blurred(model, cfg, dataset_original, dataset_blurred, filename="evalBlurred.xlsx", saveDir='./',
                            evaluate_original=True, eval_data_original="", use_gpu=True, saveJson=True,
                            metrics=['accuracy', 'precision', 'recall', 'f1_score', 'support'],**kwargs):
    """Evaluates the original dataset and the blurred dataset and saves the resulting
    metrics into an excel file.

    :param model: Classification Model to be evaluated
    :param cfg: Config Object used for generating data_loader
    :param dataset_original: The original dataset
    :param dataset_blurred: The blurred dataset
    :param filename: Name of the resulting excel file
    :param saveDir: Path to directory where excel file will be saved to
    :param evaluate_original: Whether to evaluate the original dataset
    :param eval_data_original: Path to file containing results for original dataset. 
        Only relevant if evaluate_original=False. File must be a .json
    :param use_gpu: Compute evaluations on GPU. Otherwise CPU will be used(MUCH SLOWER).
    :param saveJson: Save evaluation results into json files.
    :param metrics: List of metrics for which the datasets will be evaluated against.
    """
    filePath = os.path.join(saveDir, filename)
    additionalEvals = True
    # Remove accuracy since that will get top-1 and top-5
    df = prepare_dataframe(metrics)
    if use_gpu:
        if isinstance(model, MMDataParallel):
            print('Model already on GPU')
        else:
            print('Evaluating Model on GPU')
            model = MMDataParallel(model, device_ids=[0])
    else:
        print("Evaluating Model on CPU")
    if evaluate_original:
        print('Computing Evaluation Metrics for Original Dataset')
        df = run_evaluation(dataset=dataset_original, cfg=cfg, model=model, metrics=metrics, saveJson=saveJson,
                            saveDir=saveDir, df=df, fileName='eval_results_original')
    else:
        if eval_data_original:
            print('Using given evaluation data of original model.')
            with open(eval_data_original, 'r') as f:
                if eval_data_original[-5:] == '.json':
                    print(f'Loading data from Json {eval_data_original}')
                    eval_results_original = json.load(f)
                else:
                    raise TypeError(f'Cannot load data from {eval_data_original}. Supported types are .json')
                df = pandas.concat((df, pandas.DataFrame.from_records([eval_results_original], index=['Evaluation_Original'])))
        else:
            print('No original data was generated or specified. Only blurred results will be saved nothing further.')
            additionalEvals = False

    print('Computing Evaluation Metrics for Blurred Dataset')
    df = run_evaluation(dataset=dataset_blurred, cfg=cfg, model=model, metrics=metrics, saveJson=saveJson,
                            saveDir=saveDir, df=df, fileName='eval_results_blurred')
    if additionalEvals:
        print('Add total Change and improvement of original over blurred')
        df = pandas.concat((df, pandas.DataFrame.from_records([df.loc['Evaluation_Original'] - df.loc['Evaluation_Blurred']], index=['Advantage_Original_over_Blurred'])))
        df = pandas.concat((df, pandas.DataFrame.from_records([abs(df.loc['Evaluation_Original'] - df.loc['Evaluation_Blurred'])], index=['Absolute_Difference_Original_Blurred'])))

    if filePath[-5:] != '.xlsx':
        filePath = filePath + '.xlsx'
    Path(os.path.dirname(filePath)).mkdir(parents=True, exist_ok=True)

    print(f'Saving evaluation results to {filePath}')
    df.to_excel(filePath)

def get_model_and_dataset(imgRoot, cfg, checkpoint, annfile, use_gpu=True, saveDir='./', **kwargs):
    """
    Loads the model and checkpoint from the given paths to config and checkpoint.
    Additionally loads the dataset for the filtered samples from the given imgRoot.
    The loaded cfg object is returned for later reference.
    The Path to the filtered annfile that is created is returned for later deletion.

    :param imgRoot: Directory of where sample data lies.
    :param cfg: Path to config for Classifier
    :param checkpoint: Path to Checkpoint for Classifier.
    :param annfile: Path to original annotation file.
    :param saveDir: Path where filtered annfile will be saved to.
    :param use_gpu: Use GPU to evaluate the model.

    :return: model, dataset, cfg, filteredAnnfilePath
    """
    imgNames = get_samples(imgRoot=imgRoot, annfile=annfile, **kwargs)
    filteredAnnfilePath = generate_filtered_annfile(annfile=annfile, imgNames=imgNames, saveDir=saveDir)

    cfg = Config.fromfile(cfg)
    cfg.data.test['ann_file'] = filteredAnnfilePath
    cfg.data.test['data_prefix'] = imgRoot
    dataset = build_dataset(cfg.data.test)
    model = build_classifier(cfg.model)
    load_checkpoint(model, checkpoint)
    if use_gpu:
        model = MMDataParallel(model, device_ids=[0])
    return model, dataset, cfg, filteredAnnfilePath

def prepare_dataframe(metrics):
    """
    Prepares the pandas Dataframe object that contains all metrics as columns.
    If accuracy is in the given metrics list it will be replaced by accuracy_top-1/_top-5

    :param metrics: list of metrics corresponding to columns
    :type metrics: list(str)

    :return df
    """
    cols = [metric for metric in metrics if metric != 'accuracy'] 
    if 'accuracy' in metrics:
        cols = cols + ['accuracy_top-1', 'accuracy_top-5']
    df = pandas.DataFrame(columns=cols)
    return df

def run_evaluation(dataset, cfg, model, metrics, saveJson=False, saveDir='./', fileName='eval_results',  df=None):
    """
    Runs the evaluation on the given Model with the same dataloader configuration
    as the cfg object specifies.
    The results will be written into the given dataframe object.
    If no dataframe is given a new one will be created.
    If specified the results will be saved in a json file.
    The dataframe containing the results will be returned.

    :param dataset: Dataset to be evaluated
    :param cfg: Loaded Config object
    :param model: Loaded model to be evaluted
    :param metrics: List of metrics to be evaluted
    :type metrics: List(str)
    :param df: dataframe object. If not specified a new one will be created.
    :type df: None or pandas.Dataframe
    :param saveJson: Save an additional Json file for the results
    :type saveJson: bool
    :param saveDir: Directory where the json file will be saved to.
    :type saveDir: str
    :param fileName: Name of the json file. Defaults to 'eval_results'
    :type fileName: str

    :return df
    """
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=cfg.data.samples_per_gpu, 
            workers_per_gpu=cfg.data.workers_per_gpu,
            shuffle=False)
    results = single_gpu_test(model, data_loader)
    print("") # New line after progress bar
    eval_results = dataset.evaluate(
        results=results,
        metric=metrics
    )
    """
    dictionary of ['accuracy_top-1', 'accuracy_top-5', 'support', 'precision', 'recall', 'f1_score']
    (Well depending on metrics parameter)
    """
    if df is not None:
        df = pandas.concat((df, pandas.DataFrame.from_records([eval_results], index=['Evaluation_Original'])))
    else: 
        df = pandas.DataFrame.from_records([eval_results], index=['Evaluation_Original'])
    if saveJson:
        save_json(eval_results, save_dir=saveDir, fileName=fileName)
    return df


def get_eval_metrics(modelCfg, modelCheckpoint, imgRoot, annfile, metrics=['accuracy', 'precision', 'recall', 'f1_score', 'support'], 
                    use_gpu=True, saveDir='./', fileName='eval_results', **kwargs):
    """
    Evaluted the model on the specified metrics given by the path to config and checkpoint on the dataset based 
    on the specified annotation file / dataClasses from the imgRoot directory.
    The results will be saved as a json file.
    """
    print('Evaluating metrics on model.')
    model, dataset, cfg, filteredAnnfilePath = get_model_and_dataset(imgRoot, modelCfg, modelCheckpoint, annfile, use_gpu=use_gpu, **kwargs)
    df = prepare_dataframe(metrics)
    run_evaluation(dataset, cfg, model, metrics, saveJson=True, saveDir=saveDir, fileName=fileName, df=df)
    os.remove(filteredAnnfilePath)