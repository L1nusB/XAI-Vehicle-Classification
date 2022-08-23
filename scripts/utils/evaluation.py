import numpy as np
import pandas
import json
import os
from pathlib import Path

from mmcls.apis.test import single_gpu_test
from mmcls.datasets.builder import build_dataloader
from mmcv.parallel import MMDataParallel

from .io import save_json

def compare_original_blurred(model, cfg, dataset_original, dataset_blurred, filename="evalBlurred.xlsx", saveDir='./',
                            evaluate_original=True, eval_data_original="", use_gpu=True, save_json=True,
                            metrics=['accuracy', 'precision', 'recall', 'f1_score', 'support']):
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
    :param save_json: Save evaluation results into json files.
    :param metrics: List of metrics for which the datasets will be evaluated against.
    """
    filePath = os.path.join(saveDir, filename)
    additionalEvals = True
    # Remove accuracy since that will get top-1 and top-5
    cols = [metric for metric in metrics if metric != 'accuracy'] 
    if 'accuracy' in metrics:
        cols = cols + ['accuracy_top-1', 'accuracy_top-5']
    df = pandas.DataFrame(columns=cols)
    if use_gpu:
        print('Evaluating Model on GPU')
        model = MMDataParallel(model, device_ids=[0])
    else:
        print("Evaluating Model on CPU")
    if evaluate_original:
        print('Computing Evaluation Metrics for Original Dataset')
        data_loader = build_dataloader(
            dataset_original,
            samples_per_gpu=cfg.data.samples_per_gpu, 
            workers_per_gpu=cfg.data.workers_per_gpu,
            shuffle=False)
        results = single_gpu_test(model, data_loader)
        print("") # New line after progress bar
        eval_results_original = dataset_original.evaluate(
            results=results,
            metric=metrics
        )
        """
        dictionary of ['accuracy_top-1', 'accuracy_top-5', 'support', 'precision', 'recall', 'f1_score']
        (Well depending on metrics parameter)
        """
        pandas.concat((df, pandas.DataFrame.from_records([eval_results_original], index=['Evaluation_Original'])))
        if save_json:
            save_json(eval_results_original, save_dir=saveDir, fileName='eval_results_original')
    else:
        if eval_data_original:
            with open(eval_data_original, 'r') as f:
                if eval_data_original[-5:] == '.json':
                    print(f'Loading data from Json {eval_data_original}')
                    eval_data_original = json.load(f)
                else:
                    raise TypeError(f'Cannot load data from {eval_data_original}. Supported types are .json')
                df = pandas.concat((df, pandas.DataFrame.from_records([eval_results_original], index=['Evaluation_Original'])))
        else:
            print('No original data was generated or specified. Only blurred results will be saved nothing further.')
            additionalEvals = False

    print('Computing Evaluation Metrics for Blurred Dataset')
    data_loader = build_dataloader(
        dataset_blurred,
        samples_per_gpu=cfg.data.samples_per_gpu, 
        workers_per_gpu=cfg.data.workers_per_gpu,
        shuffle=False)
    results = single_gpu_test(model, data_loader)
    print("") # New line after progress bar
    eval_results_blurred = dataset_blurred.evaluate(
        results=results,
        metric=metrics
    )
    """
    dictionary of ['accuracy_top-1', 'accuracy_top-5', 'support', 'precision', 'recall', 'f1_score']
    (Well depending on metrics parameter)
    """
    df = pandas.concat((df, pandas.DataFrame.from_records([eval_results_blurred], index=['Evaluation_Blurred'])))
    if save_json:
            save_json(eval_results_blurred, save_dir=saveDir, fileName='eval_results_blurred')
    if additionalEvals:
        print('Add total Change and improvement of original over blurred')
        df = pandas.concat((df, pandas.DataFrame.from_records([df.loc['Evaluation_Original'] - df.loc['Evaluation_Blurred']], index=['Advantage_Original_over_Blurred'])))
        df = pandas.concat((df, pandas.DataFrame.from_records([abs(df.loc['Evaluation_Original'] - df.loc['Evaluation_Blurred'])], index=['Absolute_Difference_Original_Blurred'])))

    if filePath[-5:] != '.xlsx':
        filePath = filePath + '.xlsx'
    Path(os.path.dirname(filePath)).mkdir(parents=True, exist_ok=True)

    print(f'Saving evaluation results to {filePath}')
    df.to_excel(filePath)