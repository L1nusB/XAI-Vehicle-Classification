import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def highlight_dominants(bars, dominantMask, color='red'):
    for index,dominant in enumerate(dominantMask):
        if dominant:
            bars[index].set_color(color)


def add_text(bars, ax, format, adjust_ypos=False, dominantMask=None, valueModifier=1, **kwargs):
    for index, bar in enumerate(bars):
        height = bar.get_height()
        ypos = height
        if adjust_ypos:
            if dominantMask is not None and dominantMask[index]:
                ypos /= 2
            else:
                ypos += 0.01
        ax.text(bar.get_x()+bar.get_width()/2.0, ypos, f'{height*valueModifier:{format}}', ha='center', va='bottom' , **kwargs)


def plot_bar(ax, x_ticks=None, data=None, dominantMask=None, hightlightDominant=True, hightlightColor='red', x_tick_labels=None, 
            bars=None, addText=True,  format='.1%', keep_x_ticks=False, increase_ylim_scale=1, **kwargs):
    """Creates a bar plot in the given axis from the given data. Customizations can be done
    in the x_ticks, labels, etc.

    :param ax: Axis in which the plot will be created
    :param x_ticks: Position of x_ticks under which the plot will create the bars. If x_tick_labels is not specified also specifies the labels
    :param data: The data which generates the bars
    :param dominantMask: Binary mask which bars to highlight if highlightDominant is specified.
    :param hightlightDominant: Whether to highlight dominant bars given by dominantMask
    :param hightlightColor: (default 'red') Color of the dominant highlights.
    :param x_tick_labels: Labels used on the x-axis
    :param bars: Already bar plot object which will be modified. If not specified one will be generated from the given data.
    :param addText: Whether to add Text onto the bars with the given format.
    :param format: Format under which the text will be added to the bars. The text indicates the height of the bar.
    :param keep_x_ticks: Keep the x_ticks that are already set or are set by default.
    :param increase_ylim_scale: Increase the y-axis limit by the given scale.

    Optional more parameters can be passed using kwargs. 
    For additional arguments for the generation of the bar plot they must be prefixed with 'bar' and be 
    usable from the matplotlib.pyplot .bar function call
    For additional arguments for adding text they must be prefixed with 'text'
    Possible options are:
    adjust_ypos:(boolean defaults to False) If set dominant bars will have their text placed into the center of the bar.
            and all other raised by .01 to move them above the bar
    valueModifier:(int defaults to 1) Multiplies the displayed value (height * valueModifier)
    Other arguments used by pyplot .text function
    """
    if bars is None:
        barkwargs = {key[len('bar'):]:value for key,value in kwargs.items() if key.startswith('bar')}
        bars = ax.bar(x_ticks, data, **barkwargs)
    # ticks_loc = ax.get_xticks()
    # ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    # if x_tick_labels is not None:
    #     ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
    # else:
    #     ax.set_xticklabels(x_ticks, rotation=45, ha="right")
    if x_tick_labels is not None:
        if not keep_x_ticks:
            ax.set_xticks(np.arange(len(x_tick_labels)))
        ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
    else:
        if not keep_x_ticks:
            ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, rotation=45, ha="right")
    if hightlightDominant and dominantMask is not None:
        highlight_dominants(bars, dominantMask, hightlightColor)
        
    if addText:
        textkwargs = {key[len('text'):]:value for key,value in kwargs.items() if key.startswith('text')}
        add_text(bars, ax, format, dominantMask=dominantMask, **textkwargs)

    if increase_ylim_scale != 1:
        ax.set_ylim(top=ax.get_ylim()[1] * increase_ylim_scale)


def plot_errorbar(ax, x_ticks, meanData, stdData, x_tick_labels=None, addText=True,  format='.2f', **kwargs):
    """Creates a errorbar plot in the given axis from the given data. Customizations can be done
    in the x_ticks, labels, etc.

    :param ax: Axis in which the plot will be created
    :param x_ticks: Position of x_ticks under which the plot will create the bars. If x_tick_labels is not specified also specifies the labels
    :param meanData: The mean Data which generates the bars
    :param stdData: The standard Deviation Data which generates the bars
    :param x_tick_labels: Labels used on the x-axis
    :param addText: Whether to add Text onto the bars with the given format.
    :param format: Format under which the text will be added to the bars. The text indicates the height of the bar.

    Optional more parameters can be passed using kwargs. 
    For additional arguments for the generation of the bar plot they must be prefixed with 'bar' and be 
    usable from the matplotlib.pyplot .bar function call
    For additional arguments for adding text they must be prefixed with 'text'
    Possible options are:
    adjust_ypos:(boolean defaults to False) If set dominant bars will have their text placed into the center of the bar.
            and all other raised by .01 to move them above the bar
    valueModifier:(int defaults to 1) Multiplies the displayed, value (height * valueModifier)
    Other arguments used by pyplot .text function
    """
    errorbarkwargs = {key[len('errorbar'):]:value for key,value in kwargs.items() if key.startswith('errorbar')}
    # Set default Formatting for errorplot
    if 'fmt' not in errorbarkwargs:
        errorbarkwargs['fmt'] = 'b_'
    if 'ecolor' not in errorbarkwargs:
        errorbarkwargs['ecolor'] = 'r'
    if 'capsize' not in errorbarkwargs:
        errorbarkwargs['capsize'] = 3
    if 'mew' not in errorbarkwargs:
        errorbarkwargs['mew'] = 2
    if 'ms' not in errorbarkwargs:
        errorbarkwargs['ms'] = 10
    for i in range(len(meanData)):
        ax.errorbar(x_ticks[i], meanData[i], yerr=stdData[i], **errorbarkwargs)
    # Add Text afterwards because only now is the height of the plot known
    up = ax.get_ylim()[1] / 50
    ax.set_ylim(top=ax.get_ylim()[1] + up)
    if addText:
        for i in range(len(meanData)):
            ax.text(i, meanData[i] + stdData[i] + ax.get_ylim()[1] / 50, f'\u03BC={meanData[i]:{format}} \n \u03C3={stdData[i]:{format}}', fontsize=8,ha='center')
    if x_tick_labels is not None:
        ax.set_xticks(np.arange(len(x_tick_labels)))
        ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
    else:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, rotation=45, ha="right")


def plot_compare_models(df, x_index='segments', y_index='Activations', hue_index='Model', palette='Set1',
                     show_ylabel=True, ax=None, hide_legend=False):
    """Plots the Activations of multiple models into one graph for comparison.

    Args:
        df (pandas.DataFrame): Dataframe with all data
        x_index (str): Column name for the x-tick labels
        y_index (str): Column name for the data
        hue_index (str): Column name to differentiate between models
        palette (str): Name of the palette to be used.
        ax (Axis): If given this axis will be used for plotting.
        hide_legend (bool): Hide the legend.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10,5))
    ax = sns.barplot(data=df, x=x_index, y=y_index, hue=hue_index, palette=sns.color_palette(palette), ax=ax)
    #sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    if hide_legend:
        ax.legend_.remove()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set(xlabel=None)
    if show_ylabel == False:
        ax.set(ylabel=None)
