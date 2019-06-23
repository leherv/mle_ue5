package mle.leher.viktor;

import lombok.Getter;

/**
 * Kombineirt mehere Neuronen zu einer Schicht die voneinander unabhängig mit den selben Eingangswerten ausgerechnet werden.
 * Einziger unterschied zwischen den Neuronen in einer Schicht sind die Gewichte.
 * Ein Layer hat somit einen mehrdimensionalen Eingang, der bei allen Neuronen hinein geschrieben wird, und einen mehrdimensionalen Ausgang, wobei ein Neuron jeweils einer Dimension entspricht.
 */
public class Layer {

    @Getter
    private Sigmoid[] nodes;
    @Getter
    private int size;
    private int prevSize;

    /**
     * Erzeugt einen Layer und befüllt ihn mit Sigmoid Neuronen
     *
     * @param size     ist die größe des Layers
     * @param prevSize ist die größe des vorhergegangenen Layers
     */

    Layer(int size, int prevSize) {
        this.size = size;
        this.prevSize = prevSize;
        nodes = new Sigmoid[size];
        for (int i = 0; i < size; i++) {
            nodes[i] = new Sigmoid(prevSize);
        }
    }


    /**
     * Berechnet für alle Neuronen im Layer <code>feedforward(input)</code>
     * und setzt einen neuen Array mit den einzelnen Ergebnissen zusammen
     *
     * @param input        der Eingabevektor des Layers
     * @param isStartLayer sollet es sich um einen Start layer handeln wird nichts gemacht
     * @return den Ausgabevektor des Layers, kann als Eingabe für einen nächsten Layer verwendet werden, oder als Ergebnis eines neuronalen Netzes
     */
    double[] feedForward(double[] input, boolean isStartLayer) {
        double[] output = new double[size];
        if (isStartLayer) {
            for (int i = 0; i < nodes.length; i++) {
                output[i] = nodes[i].feedForward(input[i]);
            }
        } else {
            for (int i = 0; i < nodes.length; i++) {
                output[i] = nodes[i].feedForward(input);
            }
        }
        return output;
    }

    @Override
    public String toString() {
        String s = "";
        for (Sigmoid node : nodes) {
            s += "\n" + node.toString();
        }
        return s;
    }

    public Sigmoid getNode(int i){
        return this.nodes[i];
    }
}
