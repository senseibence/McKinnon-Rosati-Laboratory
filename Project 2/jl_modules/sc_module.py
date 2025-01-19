import pandas as pd
import numpy as np
from seaborn import blend_palette
import matplotlib.pyplot as plt
import matplotlib.colors as mpcl

# Wes Anderson pastels for clusters on UMAP plots
wes = ['#8DD3C7','#FFFFB3','#BEBADA','#FB8072','#80B1D3','#FDB462','#B3DE69','#FCCDE5','#D9D9D9','#BC80BD','#CCEBC5','#FFED6F',
            '#77AADD','#EE8866','#EEDD88','#FFAABB','#99DDFF','#44BB99','#BBCC33','#999900','#BC9CC9','#295A8B',
             '#B14D2C','#C2AA36','#D4788A','#3E8AB1','#158364','#737A38','#4E4E00','#835794','#EDF761','#4D8CC4',
             '#C7221F','#6EB288','#74008B','#B3FD33','#D7AE3E','#526FBD','#5DA8A2','#9BFDF8','#A473B0','#E59637',
             '#C2C2C2','#521913','#E04E29','#529CB4','#8B201A','#8FBC65','#FDB462','#BEBADA','#FF5A48','#FCCDE5',
             '#BC80BD','#CCEBC5','#6A3D9A','#9B7DB7','#CAFFFE','#1F78B4','#0178FF','#FF6DB6','#FFF5C3',
             '#B8787A','#33A02C','#52BFF8','#E89976','#FF7F00','#FFECD1','#19C2C1','#FFFF50','#BDA985','#5BF1FC',
             '#98B299','#D1F8B0','#51BCAE','#EBFFC5','#C1FFD2','#E0FFB3','#79BAFF','#FDBF6F']
# sb.palplot(my_palette) # to plot palette

# color for gene expression on UMAP plots
fire = blend_palette(["#F2F2F2","#F2F2F2","#FDD5A9","#F29554","#DE3233","#882424","#641B1B"], 255, as_cmap=True)
#fire # to plot color gradient


# useful for check cell type dictionaries
def validate_dict(d, all_genes):
    msg = ''

    # get list of all marker genes
    gene_list1 = []
    for g in list(d.values()):
        for item in g:
            gene_list1.append(item)

    # check for duplicate genes
    dup = {x for x in gene_list1 if gene_list1.count(x) > 1}
    if len(dup) > 0:
        msg = msg + 'WARNING: duplicate genes present: ' + ' '.join(dup) + '\n\n'

    # remove genes that are not expressed
    keys_without_genes = []
    for key in d:
        expressed = [x for x in d[key] if x in list(all_genes)]
        if expressed:
            d[key] = expressed
        else:
            keys_without_genes.append(key)
            msg = msg + 'WARNING: This cell type is not represented: ' + key + '\n\n'
    # remove keys without any genes
    for key in keys_without_genes:
        del d[key]

    # get list of all marker genes after removing unexpressed
    gene_list2 = []
    for g in list(d.values()):
        for item in g:
            gene_list2.append(item)

    # find genes not expressed
    if len(gene_list1) > len(gene_list2):
        msg = msg + 'WARNING: Genes not expressed = ' + ' '.join(list(set(gene_list1).difference(gene_list2)))

    return d, msg

# general purpose graphing routine for plotting PCA results
def pcaplot(pca_xy, prin_comps = ['PC1', 'PC2'], eigenvectors = pd.DataFrame(), size = (8,5), legend = False, labels = False):

    # unpack PCA df and plot xy values
    observations = list(pca_xy.index)
    # get limits
    max1 = np.abs(np.max(pca_xy[prin_comps[0]].to_numpy()))
    max2 = np.abs(np.max(pca_xy[prin_comps[1]].to_numpy()))
    if max1 > max2:
        limit = max1 + 0.1*max1
    else:
        limit = max2 + 0.1*max2

    plt.figure(figsize=size)

    # get n colors
    n = pca_xy.shape[0]
    cmap = plt.get_cmap('rainbow', n)
    colors = []
    for i in range(cmap.N):
        rgba = cmap(i)
        colors.append(mpcl.rgb2hex(rgba))
    # plot data points
    for obs, color in zip(observations, colors):
      indicesToKeep = pca_xy.index == obs
      plt.scatter(pca_xy.loc[indicesToKeep, prin_comps[0]], pca_xy.loc[indicesToKeep, prin_comps[1]], label=color, c = color, s = 50)

    # create legend
    if legend:
        plt.legend(observations, loc='center left', bbox_to_anchor=(1, 0.5))
        ax = plt.gca()
        leg = ax.get_legend()
        for i in range(n):
            leg.legendHandles[i].set_color(colors[i])

    # add labels
    if labels:
        for i, txt in enumerate(pca_xy.index.tolist()):
            plt.annotate('  '+txt, (pca_xy[prin_comps[0]][i], pca_xy[prin_comps[1]][i]))

    # unpack eigenvectors and plot
    if len(eigenvectors.index) != 0:
        n = eigenvectors.shape[0] # number of variables
        x = list(eigenvectors[prin_comps[0]])
        y = list(eigenvectors[prin_comps[1]])
        features = list(eigenvectors.index)
        for i in range(n):
          plt.arrow(0, 0, x[i], y[i], head_width=0.2, head_length=0.2)
          plt.text(x[i]*1.4, y[i]*1.4, features[i], color = 'k', ha = 'center', va = 'center',fontsize=10)

    plt.xlabel(prin_comps[0], size=16)
    plt.ylabel(prin_comps[1], size=16)
    plt.grid(False)
    plt.xlim(-limit,limit)
    plt.ylim(-limit,limit)
    plt.tick_params(axis='both', which='both', labelsize=14)
    plt.show()

# plot variance for principal components
def variance_plot(var_exp):
    # select number of eigenvalues to plot
    if len(var_exp) < 10:
        num_pcs = len(var_exp)
    else:
        num_pcs = 10
    cum_sum_exp = np.cumsum(var_exp)

    plt.bar(range(1,num_pcs+1), var_exp[0:num_pcs], alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1,num_pcs+1), cum_sum_exp[0:num_pcs], where='mid',label='Cumulative explained variance')

    plt.ylabel('Explained variance ratio', fontsize=18)
    plt.xlabel('Principal Component', fontsize=18)
    plt.xticks(np.arange(1, num_pcs+1))
    plt.ylim(top=1)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 1), prop={'size': 18})
    plt.grid(False)
    plt.show()

# useful to display multiple dataframes side by side
from IPython.display import display, HTML
def display_side_by_side(dfs:list, precisions:list, captions:list):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        precisions: list of precision to display data
        captions: list of table captions
    """
    output = ""
    for (df, precision, caption) in zip(dfs, precisions, captions):
        output += df.style.set_table_attributes("style='display:inline'").format(precision=precision)\
        .set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))
