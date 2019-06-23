package mle.leher.viktor;

/**
 * Jedes Neuronale Netz besteht aus Layern, die aus Neuronen bestehen.
 * Eingaben werden in den ersten Layer geschireben und berechnet, das Ergenbis davon wird in den 2ten Layer geschrieben, usw.
 * Die Ausgabe des Neuronalen Netzes ist die Ausgabe des letzten Layers
 */
public class Network {

    private Layer[] layers;
    private int size;

    /**
     * Erzeugt ein Neuronales Netzwerk und setzt alle Gewichte der Neuronen auf Zufallszahlen
     *
     * @param size gibt die Größe des Netzes an zb {10, 100, 10} hat 10 Neuronen im ersten Layer, 100 im 2tn und 10 im dritten
     */
    Network(int[] size) {
        layers = new Layer[size.length];
        this.size = size.length;
        for (int i = 0; i < size.length; i++) {
            if (i == 0) {
                layers[i] = new Layer(size[i], 0);
            } else {
                layers[i] = new Layer(size[i], size[i - 1]);
            }
        }
    }

    /**
     * Brechnet die Ausgabe des Netzes für input als Eingabe
     *
     * @param input
     * @return
     */
    double[] feedForward(double[] input) {
        for (int i = 0; i < layers.length; i++) {
            if (i == 0) {
                input = layers[i].feedForward(input, true);
            } else {
                input = layers[i].feedForward(input, false);
            }
        }
        return input;
    }

    @Override
    public String toString() {
        String s = "";
        for (Layer layer : layers) {
            s += "\n------------------------------------------------------\n" + layer.toString();
        }

        return s;
    }

    Layer getLayer(int i) {
        return this.layers[i];
    }

    Layer[] getLayers() {
        return layers;
    }

    public void setLayers(Layer[] layers) {
        this.layers = layers;
    }

    int getSize() {
        return this.size;
    }

}
