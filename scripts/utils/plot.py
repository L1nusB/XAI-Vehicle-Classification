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


def plot_bar(ax, x_ticks=None, data=None, dominantMask=None, hightlightDominant=True, x_tick_labels=None, 
            bars=None, addText=True,  format='.1%', **kwargs):
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
        highlight_dominants(bars, dominantMask)

    if addText:
        textkwargs = {key[len('text'):]:value for key,value in kwargs.items() if key.startswith('text')}
        add_text(bars, ax, format, dominantMask=dominantMask, **textkwargs)
    