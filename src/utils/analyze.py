"""
Given a file, output the Mean and Standard Deviation of the values in the file.
"""
import sys
from os.path import dirname
sys.path.append(dirname(__name__))
sys.path.append('../')

import numpy as np
from args import parse_args
import matplotlib.pyplot as plt
import statistics
from math import sqrt

plt.rcParams['font.serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams.update({'font.size': 18})

def plot_confidence_interval(x, values, z=1.96, color='#8E8DC8', horizontal_line_width=0.25):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, 'o', color='#8E8DC8')

    return mean, confidence_interval

def main():
    args = parse_args()
    if args.experiment=="roc-score":
        file_path = "results/{}-{}-roc-score.txt".format(args.dataset, args.arch)
        with open(file_path, "r") as f:
            while True:
                line = f.readline()
                if line == "VertexSerum Attention\n":
                    print(line)
                    f.readline()
                    data = f.readline().strip()
                    data = np.fromstring(data, sep=',')
                    print(np.mean(data), np.std(data))
                if line == "VertexSerum MLP\n":
                    print(line)
                    f.readline()
                    data = f.readline().strip()
                    data = np.fromstring(data, sep=',')
                    print(np.mean(data), np.std(data))
                if line == "Link Steal Attention\n":
                    print(line)
                    f.readline()
                    data = f.readline().strip()
                    data = np.fromstring(data, sep=',')
                    print(np.mean(data), np.std(data))
                if line == "Link Steal MLP\n":
                    print(line)
                    f.readline()
                    data = f.readline().strip()
                    data = np.fromstring(data, sep=',')
                    print(np.mean(data), np.std(data))
                if line == "":
                    break
    elif args.experiment=="oversmooth":
        file_path = "results/{}-{}-oversmooth.txt".format(args.dataset, args.arch)
        with open(file_path, "r") as f:
            result = []
            while True:
                line = f.readline()
                data = line.strip()
                data = np.fromstring(data, sep=',')
                if data.size>0:
                    result.append(data)
                if line == "Train accuracy\n":
                    line = f.readline()
                    train_acc = np.fromstring(line, sep=',')
                    print(np.mean(train_acc), np.std(train_acc))
                if line == "Test accuracy\n":
                    line = f.readline()
                    test_acc = np.fromstring(line, sep=',')
                    print(np.mean(test_acc), np.std(test_acc))
                if line == "":
                    break
            result = np.array(result)
            print(np.median(result, axis=0))
        
        # visualize the relation between number of layers, training accuracy, testing accuracy and attack AUC.
        fig, ax1 = plt.subplots(figsize=[10, 4])
        # visualize the median of attack AUC
        line1 = plt.plot(np.arange(1, 11), np.mean(result, axis=0), color='#2E2C69', lw=4, label="Attack AUC")
        # visualize the confidence interval
        for x in np.arange(1, 11):
            plot_confidence_interval(x, result[:, x-1])
        plt.ylabel('AUC')
        plt.xlabel('Number of layers')
        ax2 = ax1.twinx()
        # visualize the training accuracy
        line2 = ax2.plot(np.arange(1, 11), train_acc, ls="--", marker="o", lw=2, color='#D8A398', label="Train Acc.")
        # visualize the testing accuracy
        line3 = ax2.plot(np.arange(1, 11), test_acc, ls="--", marker="o", lw=2, color='#F7E2DB', label="Test Acc.")
        plt.xticks(np.arange(1, 11))
        plt.ylabel('Acc.')
        plt.xlabel('Number of layers')
        lines = line1 + line2 + line3 
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3)
        plt.grid()
        plt.tight_layout()
        plt.savefig("results/oversmooth.pdf")
    if args.experiment=='online':
        file_path = "results/{}-{}-{}.txt".format(args.dataset, args.arch, args.experiment)
        with open(file_path, "r") as f:
            result = []
            while True:
                line = f.readline()
                data = line.strip()
                data = np.fromstring(data, sep=',')
                if data.size>0:
                    result.append(data)
                if line == "":
                    break
            result = np.array(result)
            print(np.mean(result, axis=1))
        
        # visualize the relationship between different injection round and the attack successful rate
        fig, ax1 = plt.subplots(figsize=[10, 4])
        # visualize the attack successful rate
        line1 = plt.plot(np.arange(1, 9), np.mean(result, axis=1), color="#2E2C69" ,lw=4, label="Online AUC")
        # visualize the confidence interval
        for x in np.arange(1, 9):
            plot_confidence_interval(x, result[x-1, :])
        plt.hlines(y=0.954, xmin=1, xmax=8, color='#D8A398', ls='--', lw=4, label='Offline AUC')
        plt.legend()
        plt.ylabel('AUC')
        plt.xlabel('Adversarial Batch Index')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2)
        plt.grid()
        plt.tight_layout()
        plt.savefig("results/online.pdf")
    if args.experiment=='shadow':
        data = {}
        for arch in ["gat", "sage", "gcn"]:
            data[arch] = {}
            file_path = "results/{}-{}-{}.txt".format(args.dataset, arch, args.experiment)
            with open(file_path, "r") as f:
                while True:
                    line = f.readline()
                    # data = np.fromstring(data, sep=',')
                    if line == "shadow model is gat\n":
                        line = f.readline()
                        line = f.readline()
                        gat = line.strip()
                        gat = np.fromstring(gat, sep=',')
                    if line == "shadow model is gcn\n":
                        line = f.readline()
                        line = f.readline()
                        gcn = line.strip()
                        gcn = np.fromstring(gcn, sep=',')
                    if line == "shadow model is sage\n":
                        line = f.readline()
                        line = f.readline()
                        sage = line.strip()
                        sage = np.fromstring(sage, sep=',')
                    if line == "":
                        break
                data[arch]["shadow-gat"] = gat
                data[arch]["shadow-gcn"] = gcn
                data[arch]["shadow-sage"] = sage
        # convert the dictionary to database pandas
        import pandas as pd 
        import seaborn as sns
        df = pd.DataFrame.from_dict(data, orient='index')
        df = pd.melt(df.reset_index(), id_vars='index')
        df.columns = ['x', 'group', 'value']
        df['mean'] = np.asarray(val.mean(axis=0) for val in df.value.values)
        df['std'] = np.asarray(val.std(axis=0) for val in df.value.values)
        fig, ax = plt.subplots(figsize=[9, 4.5])
        custom_palette = sns.color_palette([ '#8E8DC8', '#F7E2DB' ,'#D8A398', '#2E2C69'])
        ax = sns.barplot(data=df, x='x', y='mean', hue='group', edgecolor='black', palette=custom_palette, width=0.4)
        
        hatches = ['-', '+', 'x']

        # for i, patch in enumerate(ax.patches):
        #     # calculate the index of the grouping variable
        #     idx = i // len(df['x'].unique())
        #     # set the hatch pattern based on the index
        #     patch.set_hatch(hatches[idx])

        x_coords = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
        y_coords = [p.get_height() for p in ax.patches]

        plt.errorbar(x=x_coords, y=y_coords, yerr=df["std"], fmt='o', color='black',
             ecolor='k', elinewidth=3, capsize=5)
        # plt.grid()
        ax.set_xlabel('Victim Architecture', labelpad=10, position=(0.5, -0.15))
        ax.set_ylabel('AUC', labelpad=10, position=(-0.15, 0.5), rotation=90)
        ax.set_ylim(0.6, 1)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
        plt.tight_layout()
        plt.savefig("results/shadow.pdf")
    if args.experiment=='regularizer':
        print("Evaluate the impact of different regularizer terms on the attack performance.")
        file_path = "results/{}-{}-{}.txt".format(args.dataset, args.arch, args.experiment)
        with open(file_path, "r") as f:
            result = []
            while True:
                line = f.readline()
                data = line.strip()
                data = np.fromstring(data, sep=',')
                if data.size>0:
                    print(data.mean())
                else:
                    print(line)
                if line == "":
                    break

if __name__ == '__main__':
    main()
