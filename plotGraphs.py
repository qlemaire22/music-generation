import csv
import matplotlib.pyplot as plt
import argparse

def plot(run_name):
    with open('results/' + run_name + '/train_log.csv', 'r') as csvfile:
        epochs = []
        loss = []
        val_loss = []
        spamreader = csv.reader(csvfile, delimiter=';')
        i = 0
        for row in spamreader:
            if i != 0:
                if len(row) == 3:
                    epochs.append(int(row[0]))
                    loss.append(float(row[1]))
                    val_loss.append(float(row[2]))
                else:
                    epochs.append(int(row[0]))
                    loss.append(float(row[1]))
            i += 1

        loss_plot = plt.plot(epochs, loss, label="training loss")
        if val_loss != []:
            val_loss_plot = plt.plot(epochs, val_loss, label="validation loss")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('results/' + run_name + 'graph.png')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', default="run1",
                        help="str, the name of the run.", type=str)

    args = parser.parse_args()

    plot(args.run_name)
