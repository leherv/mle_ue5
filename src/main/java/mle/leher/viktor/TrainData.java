package mle.leher.viktor;

import lombok.Getter;
import lombok.Setter;

import java.util.Random;

/**
 * Repräsentiert TestDaten für eine Neuronales Netz
 * Jeder Eintrag hat Daten, die in ein Neuronales Netz eingespeist werden und ein Label welches dem idealen Output des NNs entspricht.
 */

class TrainData {

    @Getter
    @Setter
    private double[][] data;

    @Getter
    @Setter
    private double[][] label;

    @Getter
    @Setter
    private int length;

    TrainData(boolean test) {

        if (test) {
            setData(MnistLoader.getTestImages());
            setLabel(MnistLoader.getTestLabels());
        } else {
            setData(MnistLoader.getTrainingsImages());
            setLabel(MnistLoader.getTrainingsLabels());
        }
        setLength(this.data.length);
    }

    /**
     * Mischt die TestDaten neu, sollte bei längeren Daten nicht oft verwendent werden, da sehr ineffizient
     */
    void shuffle() {
        int[] shuffleIndex = new int[length];
        for (int i = 0; i < length; i++) {
            shuffleIndex[i] = i;
        }
        shuffleArray(shuffleIndex);
        double[][] data = new double[this.data.length][this.data[1].length];
        double[][] label = new double[this.label.length][this.label[1].length];
        for (int i = 0; i < length; i++) {
            data[i] = this.data[shuffleIndex[i]];
            label[i] = this.label[shuffleIndex[i]];


        }
        this.data = data;
        this.label = label;
    }

    private void shuffleArray(int[] array) {
        int index, temp;
        Random random = new Random();
        for (int i = array.length - 1; i > 0; i--) {
            index = random.nextInt(i + 1);
            temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }

    double[] getData(int index) {
        return data[index];
    }

    double[] getLabel(int index) {
        return label[index];
    }
}
