package mle.leher.viktor;

import org.javatuples.Pair;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by dkude on 24.03.2016.
 */
public class NeuralNet {
    private double[][][] network;

    /**
     * Erstellt ein neues NN anhand einer txt Datei, die ein statischer Teil des Programmes ist
     * In dieser Datei sind alle Paramter des NNs gespeichert
     *
     * @param file ist ein double[][][] Array welcher mittels ObjectOutputStream gespeichert wurde.
     *             Wobei die Struktur [layer][Node][Weight] ist.
     */
    public NeuralNet(String file) {
        try(ObjectInputStream o = new ObjectInputStream(new FileInputStream(file))) {
            network = (double[][][]) o.readObject();
        } catch (FileNotFoundException | ClassNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            System.err.println("File not found");
        }
    }

    /**
     * @param input ist ein Array, der einem aufgefädelten Bildes entspricht, d.h   ObjekErkennung.Pixel[0][1] => ObjekErkennung.Pixel[n+1],
     *              muss 784 Werte lang sein. Entspricht einem 28*28 ObjekErkennung.Pixel großem Bild.
     * @return Ein 10 Stellen langer Array der der Klassifizierung der Ziffer entspricht.
     * d.h. es sollte immer nur ein Eintrag ungefähr 1 sein und alle anderen ungefähr null
     */
    private double[] feedForward(double[] input) {
        if (input.length == network[0].length) {
            for (int layer = 1; layer < network.length; layer++) { //start at 1 because output cant
                double[] newInput = new double[network[layer].length];
                for (int node = 0; node < network[layer].length; node++) {
                    double net = 0;
                    for (int weight = 1; weight < network[layer][node].length; weight++) {
                        net += input[weight - 1] * network[layer][node][weight];
                    }
                    net += network[layer][node][0];
                    newInput[node] = 1 / (1 + Math.exp(-net));
                }
                input = newInput;
            }
            return input;
        } else {
            return null;
        }
    }


    Pair<int[], int[]> classify(TrainData test) {
        var trueValues = Arrays.stream(test.getLabel())
                .mapToInt(this::getMaxindex)
                .toArray();

        var predictedValues = Arrays.stream(test.getData())
                .mapToInt(doubleAr -> getMaxindex(feedForward(doubleAr)))
                .toArray();

        return Pair.with(predictedValues, trueValues);
    }


    /**
     * Testet die Genauigkeit des NNs anhand einer TestDatei. Gibt die Prozent der falsch klassifizierten Ziffern an.
     *
     * @param test TrainingsData test enspricht der 10.000 stelligen Datenbank von Ziffern der Mnist-Database.
     */
    public void testAccuracy(TrainData test) {
        ArrayList<Integer> wronglyClassified = new ArrayList<>();

        for (int sample = 0; sample < test.getData().length; sample++) {
            double[] output = feedForward(test.getData(sample));
            int outputDigit = getMaxindex(output);
            int desiredDigit = getMaxindex(test.getLabel(sample));

            if (outputDigit == desiredDigit) {

            } else {
                wronglyClassified.add(sample);
            }
        }
        System.out.println((1 - (double) wronglyClassified.size() / (double) test.getData().length) * 100 + "% Accuracy");
    }


    /**
     * Gibt den index des größten Wertes im Array aus. Entspricht bei einem Ergebnis des NN der klassifizierten Ziffer.
     *
     * @param array double Array
     * @return Max ObjekErkennung.Index des Arrays
     */
    private int getMaxindex(double[] array) {
        int index = -1;
        if (array != null) {
            double biggest = Double.MIN_VALUE;
            for (int i = 0; i < array.length; i++) {
                if (array[i] > biggest) {
                    biggest = array[i];
                    index = i;
                }
            }
        }
        return index;
    }

}
