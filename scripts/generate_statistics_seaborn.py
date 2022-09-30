import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import numpy as np
import copy
import seaborn as sns
from datetime import date

from .utils.io import get_samples, save_result_figure_data, save_excel_auto_name, load_results_excel, load_result_excel_pandas_longform
from .utils.prepareData import prepareInput, prepare_generate_stats, prepare_model_comparison_dataframe
from .utils.calculations import generate_stats, accumulate_statistics, get_area_normalized_stats, get_top_k_pandas, accumulate_statistics_together
from .utils.plot import plot_bar, plot_errorbar, plot_compare_models
from .utils.model import get_wrongly_classified
from .utils.preprocessing import load_classes

from .utils.constants import EXCELCOLNAMESSTANDARD, EXCELCOLNAMESPROPORTIONAL, EXCELCOLNAMESNORMALIZED, EXCELCOLNAMESMEANSTDTOTAL, EXCELCOLNAMESMISSCLASSIFIED

def generate_model_comparison(*paths, x_index='segments', y_index='Activations', hue_index='Model',
                                types=['ResNet', 'SwinBase', 'SwinSmall'], fileExtension='.pdf',
                                fileName='modelComparison', save_dir='./', save=True, n_plots=1, hide_legend_subsequent=True,
                                dataColumnName='PercActivations', title='', **kwargs):
    """Plots a comparison between the models specified by models parameter from the excel files given through paths.

    Args:
        x_index (str): Column name for the x-tick labels
        y_index (str): Column name for the data
        hue_index (str): Column name to differentiate between models
        palette (str): Name of the palette to be used.
        types (list, optional): name of models or otherwise differentiating type. Defaults to ['ResNet', 'SwinBase', 'SwinSmall'].
        show_ylabel (bool): Show label on y-axis
        n_plots (int, default 1): If changed multiple subplots are created for each subset of paths. 
            Paths must then be given as lists.
        hide_legend_subsequent (bool, default True): Hide the legend after the first plot.

        kwargs:
        add_text: Add Text of bar heights.
        fontsize: Size of text
        show_ylabel: Show y-label in graph
        palette: Palette to draw the plot with
    """
    fig = plt.figure(figsize=(8,5),constrained_layout=True)
    grid = fig.add_gridspec(nrows=n_plots)
    if n_plots != 1:
        assert len(paths) == n_plots, f'Specified n_plots {n_plots} does not match amount of given paths (lists of paths) {len(paths)}'
        if isinstance(title, str):
            plotTitle = title
        else:
            assert isinstance(title, list) and len(title) == n_plots, f'Incorrect number of titles given: Expected where {n_plots} but given were {len(title)}'
        for i,p in enumerate(paths):
            if isinstance(title, list):
                plotTitle = title[i]
            ax = fig.add_subplot(grid[i,0])
            df = prepare_model_comparison_dataframe(p, types, newColumn=y_index, dataColumn=dataColumnName, diffColName=hue_index)
            plot_compare_models(df, x_index, y_index, hue_index, ax=ax, hide_legend=(i>0 and hide_legend_subsequent), title=plotTitle, **kwargs)
    else:
        ax = fig.add_subplot(grid[0,0])
        if isinstance(paths[0], list):
            assert len(paths) == 1, f'If n_plots is 1 only a single list or individual parameters may be passed.'
            # In case a list is still passed for a single plot
            df = prepare_model_comparison_dataframe(paths[0], types, newColumn=y_index, dataColumn=dataColumnName, diffColName=hue_index)
        else:
            df = prepare_model_comparison_dataframe(paths, types, newColumn=y_index, dataColumn=dataColumnName, diffColName=hue_index)
        plot_compare_models(df, x_index, y_index, hue_index, ax=ax, **kwargs)
    if save:
        save_result_figure_data(figure=fig, saveAdditional=False, fileExtension=fileExtension, fileName=fileName, save_dir=save_dir)

def generate_normalized(result_file, file_name='normalized', save_dir='./', n_samples=0, show_perc=True, y_axis_scale=1.2,
                         plot_normalized=True, plot_importance=True, move_legend=False):
    assert os.path.isfile(result_file), f'No such file {result_file}'

    fig = plt.figure(figsize=(15,10),constrained_layout=True)
    grid = fig.add_gridspec(ncols=1, nrows=2)
    col_names = list(EXCELCOLNAMESNORMALIZED.values())
    col_names.remove('RelativeCAMImportance')

    if plot_normalized:
        ax0 = fig.add_subplot(grid[0,0])

        ax0.set_title('Average relative CAM')

        dfNormalized = load_result_excel_pandas_longform(result_file, value_vars=col_names)

        largestActivations = get_top_k_pandas(dfNormalized, 'PercActivations')
        largestSegments = get_top_k_pandas(dfNormalized, 'PercSegmentAreas')
        largestNormalActivations = get_top_k_pandas(dfNormalized, 'PercActivationsRescaled')

        ax0 = sns.barplot(data=dfNormalized, x='segments', y='vals', hue='type', ax=ax0, palette=sns.color_palette('Paired')[::2])
        x_ticklabels = dfNormalized['segments'].unique()
        ax0.set_xticklabels(x_ticklabels, rotation=45, ha="right")
        for i,patch in enumerate(ax0.patches):
            if i < 18:
                mask = largestActivations
                col = sns.color_palette('Paired')[1]
            elif i < 36:
                mask = largestSegments
                col = sns.color_palette('Paired')[3]
            else:
                mask = largestNormalActivations
                col = sns.color_palette('Paired')[5]
            if ax0.get_xticklabels()[i % 19].get_text() in mask:
                patch.set_facecolor(col)
            if show_perc:
                ax0.text(patch.get_x()+patch.get_width()/2.0, patch.get_height() + ax0.get_ylim()[1]/70, f'{patch.get_height():.1%}', 
                    ha='center', va='bottom', rotation=90)

        if n_samples == 0:
            ax0.set(xlabel=None)
        else:
            ax0.xaxis.set_label_position('top')
            ax0.set_xlabel(f'No. samples: {n_samples}')
        ax0.set(ylabel=None)
        ax0.set_ylim(top=ax0.get_ylim()[1] * y_axis_scale)

        legend = ax0.legend_
        legend.set_title('')
        legendMap = {
            sns.color_palette('Paired')[1] : 'CAMs',
            sns.color_palette('Paired')[3] : 'Segment Areas',
            sns.color_palette('Paired')[5] : 'Normalized CAMs',
        }
        handles = [Patch(color=k, label=v) for k,v in legendMap.items()]
        ax0.legend(handles=handles)
        if move_legend:
            sns.move_legend(ax0, "upper right", bbox_to_anchor=(1, 1.25))

    if plot_importance:
        ax1 = fig.add_subplot(grid[1,0])
        ax1.set_title('CAM importance normalized by area')
        dfImportance = load_result_excel_pandas_longform(result_file, value_vars='RelativeCAMImportance')

        largestImportance = get_top_k_pandas(dfImportance, 'RelativeCAMImportance')

        ax1 = sns.barplot(data=dfImportance, x='segments', y='vals', hue='type', ax=ax1, palette=sns.color_palette("Set1")[1:])
        x_ticklabels = dfImportance['segments'].unique()
        ax1.set_xticklabels(x_ticklabels, rotation=45, ha="right")

        for i,patch in enumerate(ax1.patches):
            if ax1.get_xticklabels()[i].get_text() in largestImportance:
                patch.set_facecolor(sns.color_palette("Set1")[0])
            if show_perc:
                ax1.text(patch.get_x()+patch.get_width()/2.0, patch.get_height() + ax1.get_ylim()[1]/70, f'{patch.get_height():.1%}', 
                    ha='center', va='bottom', rotation=90)
        
        ax1.set(ylabel=None, xlabel=None)
        ax1.set_ylim(top=ax1.get_ylim()[1] * y_axis_scale)

        legend = ax1.legend_
        legend.set_title('')
        legendMap = {
            sns.color_palette('Set1')[0] : 'Relative Importance',
        }
        handles = [Patch(color=k, label=v) for k,v in legendMap.items()]
        ax1.legend(handles=handles)
        if move_legend:
            sns.move_legend(ax1, "upper right", bbox_to_anchor=(1, 1.1))

    save_result_figure_data(figure=fig, save_dir=save_dir, fileExtension='.pdf', fileName=file_name)