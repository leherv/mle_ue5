package mle.leher.viktor;

import lombok.Getter;
import lombok.Setter;

import java.util.Random;

/**
 * Repräsentiert ein Sigmoid Neuron. Ist im Grundsatz eine Abstraktion der Funktion:
 * <p>
 * f(x1, ... ,xn) = sum(x1*w1, ... , xn*wn)
 * f2 = 1/(1-(e^(-f1)))
 * <p>
 * Jedes Sigmoid Neuron hat eine Anzahl von Eingängen n und eine glechen Anzahl an sogenannten Gewichte.
 * Gewichte werden zuerst zufällig generiert und dann mittels Stochastic Gradient Decent an die Trainingsdaten angepasst.
 */

public class Sigmoid {
    @Getter
    @Setter
    private double[] weights;
    private double[] weightDiff;

    @Getter
    @Setter
    private double bias;

    @Getter
    @Setter
    private double biasDiff;

    @Getter
    private int size;

    @Getter
    @Setter
    private double output;

    @Getter
    @Setter
    private double signalError;
    private boolean isInputNode;

    /**
     * Ezeugt ein Sigmoid Neuron mit <code>size</code> Eingängen und Gewichte, die zufällig gewählt sind.
     *
     * @param size
     */

    Sigmoid(int size) {
        weights = new double[size];
        weightDiff = new double[size];
        this.size = size;
        isInputNode = size == 0;
        initialize();
    }

    private void initialize() {
        if (size != 0) {
            Random r = new Random();
            for (int i = 0; i < weights.length; i++) {
                weights[i] = r.nextGaussian() * 1 / Math.sqrt(size); //normalverteilt
            }
            bias = r.nextGaussian();
        }
    }

    /**
     * Im fall das ein Neuron in einem Input Layer steht sollen die Daten nicht verändert werden.
     * Diese Methode schreibt nur den Eingang in den Ausgang und berechnet sonst nichts.
     *
     * @param input
     * @return
     */
    double feedForward(double input) {
        output = input;
        return output;
    }

    /**
     * Berechet die obenn erwähnte Funktion.
     *
     * @param input Eingangsvektor der Funktion
     * @return Ergebnis der Funktion
     */
    double feedForward(double[] input) {
        double sum = 0;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * input[i];
        }
        sum += bias;
        output = 1 / (1 + Math.exp(-sum)); //Math.exp = e^-sum
        return output;
    }

    public String toString() {
        String s = "weights: ";
        for (double weight : weights) {
            s += weight + ", ";
        }
        s += "\nbias: " + bias;
        s += "\noutput: " + output;
        return s;
    }

    double getWeight(int index) {
        return weights[index];
    }

    void setWeight(double weight, int index) {
        this.weights[index] = weight;
    }

    double getWeightDiff(int index) {
        return weightDiff[index];
    }

    void setWeightDiff(double weightDiff, int index) {
        this.weightDiff[index] = weightDiff;
    }

}
