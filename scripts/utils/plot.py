import matplotlib.ticker as mticker


def highlight_dominants(bars, dominantMask, color='red'):
    for index,dominant in enumerate(dominantMask):
        if dominant:
            bars[index].set_color(color)


def add_text(bars, ax, format, adjust_ypos=False, dominantMask=None, valueModifier=1, **kwargs):
    for index, bar in enumerate(bars):
        height = bar.get_height()
        ypos = height
        if adjust_ypos:
            if dominantMask[index]:
                ypos /= 2
            else:
                ypos += 0.01
        ax.text(bar.get_x()+bar.get_width()/2.0, ypos, f'{height*valueModifier:{format}}', ha='center', va='bottom' , **kwargs)


def plot_bar(ax, x_ticks=None, data=None, dominantMask=None, hightlightDominant=True, hightlightColor='red', x_tick_labels=None, 
            bars=None, addText=True,  format='.1%', **kwargs):
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
    ticks_loc = ax.get_xticks()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
    else:
        ax.set_xticklabels(x_ticks, rotation=45, ha="right")

    if hightlightDominant:
        highlight_dominants(bars, dominantMask, hightlightColor)

    if addText:
        textkwargs = {key[len('text'):]:value for key,value in kwargs.items() if key.startswith('text')}
        add_text(bars, ax, format, dominantMask=dominantMask, **textkwargs)
    