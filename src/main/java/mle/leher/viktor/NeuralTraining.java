package mle.leher.viktor;

import lombok.Getter;
import lombok.Setter;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

/**
 * Diese Klasse wird zum Trainieren von Neuronalen Netzwerken verwendet.
 * Es wird mittels Stochastic Gradient Decent versucht den Fehler eines NNs zu minimieren, sodass bei allen Trainingsdaten richtige Ergenbisse ausgegeben werden.
 */
public class NeuralTraining {

    @Getter
    @Setter
    private Network network;

    @Getter
    @Setter
    private TrainData train;
    private TrainData test;

    @Getter
    @Setter
    private double learningRate;

    @Getter
    @Setter
    private double momentum;

    /**
     * Erzuegt ein neues NeuralTrain Objekt mit den spezifizierten Einstellungen.
     *
     * @param net          Das Neuronale Netz, dass trainiert werden soll
     * @param learningrate die einzelnen Gewichts änderungen werden mit diesem Wert multipliziert. Umso kleiner diese Zahl ist umso genauer wird trainiert, jedoch umso länger wird es dauern
     * @param momentum
     * @param train        die Trainingsdaten an die das Netz antrainiert wird
     * @param test         die Testdaten anhand denen das Netz validiert werden soll
     */

    public NeuralTraining(Network net, double learningrate, double momentum, TrainData train, TrainData test) {
        setNetwork(net);
        setMomentum(momentum);
        setLearningRate(learningrate);
        this.train = train;

        this.test = test;
    }

    private void backpropagate(double[] error) {

        // sets all the signal errors for every neuron on the output layer
        for (int i = 0; i < network.getLayers()[network.getLayers().length - 1].getSize(); i++) {
            Sigmoid n = network.getLayer(network.getSize() - 1).getNode(i);
            n.setSignalError(error[i] * n.getOutput() * (1 - n.getOutput()));
        }

        // sets all the signal errors for every neuron on all the hidden layers
        for (int layer = network.getSize() - 2; layer > 0; layer--) {
            for (int node = 0; node < network.getLayer(layer).getSize(); node++) {
                double signalError = 0;
                for (int nodeInPrevLayer = 0; nodeInPrevLayer < network.getLayer(layer + 1).getSize(); nodeInPrevLayer++) {
                    signalError = signalError + (network.getLayer(layer + 1).getNode(nodeInPrevLayer).getSignalError() * network.getLayer(layer + 1).getNode(nodeInPrevLayer).getWeight(node));
                }
                double output = network.getLayer(layer).getNode(node).getOutput();
                network.getLayer(layer).getNode(node).setSignalError(signalError * output * (1 - output));
            }
        }

    }

    private void updateWeigths() {
        for (int layer = 1; layer < network.getSize(); layer++) {
            for (int node = 0; node < network.getLayer(layer).getSize(); node++) {
                network.getLayer(layer).getNode(node).setBiasDiff(learningRate * network.getLayer(layer).getNode(node).getSignalError() + network.getLayer(layer).getNode(node).getBiasDiff() * momentum);
                network.getLayer(layer).getNode(node).setBias(network.getLayer(layer).getNode(node).getBias() - network.getLayer(layer).getNode(node).getBiasDiff());
                for (int weight = 0; weight < network.getLayer(layer).getNode(node).getSize(); weight++) {
                    network.getLayer(layer).getNode(node).setWeightDiff(network.getLayer(layer).getNode(node).getSignalError() * network.getLayer(layer - 1).getNode(weight).getOutput() * learningRate + network.getLayer(layer).getNode(node).getWeightDiff(weight) * momentum, weight);
                    network.getLayer(layer).getNode(node).setWeight(network.getLayer(layer).getNode(node).getWeight(weight) - network.getLayer(layer).getNode(node).getWeightDiff(weight), weight);
                }
            }
        }

    }

    Network train(long maxCycle, double errorThreshold) {
        double networkError = 1;
        for (int cycle = 0; cycle < maxCycle; cycle++) {
            System.out.println("Training cycle: " + cycle + " starting...");
//            train.shuffle();
            double[] error;
            for (int sample = 0; sample < train.getLength(); sample++) { // alle bilder (60000)
                double[] out = network.feedForward(train.getData(sample));
                error = meanSquaredErrorDiff(out, train.getLabel(sample));
                backpropagate(error);
                updateWeigths();
            }
            networkError = meanSquaredErrorOverAll();
            System.out.println("Netowrk error: " + networkError);
            if(networkError <= errorThreshold) break;
        }
        return this.network;
    }


    /**
     * Berechent den MSE für alle Inputs im TrainingsSet
     *
     * @return
     */
    public double meanSquaredErrorOverAll() {
        TrainData testData = new TrainData(true);
        double diffSum = 0;
        for (int i = 0; i < testData.getLength(); i++) {
            double[] output = network.feedForward(testData.getData(i));
            double nodeError = 0;
            for (int j = 0; j < output.length; j++) {
                nodeError += Math.pow(output[j] - testData.getLabel(i)[j], 2);
            }
            diffSum += nodeError;
        }
        return diffSum / train.getLength();
    }


    private double[] meanSquaredErrorDiff(double[] output, double[] desiredOutput) {
        return sub(output, desiredOutput);
    }

    private double[] sub(double a[], double b[]) {
        double[] sub = a.clone();
        for (int i = 0; i < a.length; i++) {
            sub[i] -= b[i];
        }
        return sub;
    }

    /**
     * Exprortiert das Neuronale Netz in eine Datei, die von der Klasse NeuralNet eingelesen werden kann
     *
     * @return double[][][]
     */
    double[][][] exportWeights() {
        double[][][] network;
        network = new double[this.network.getSize()][][];
        for (int i = 0; i < network.length; i++) {
            network[i] = new double[this.network.getLayer(i).getSize()][];
            for (int j = 0; j < network[i].length; j++) {
                network[i][j] = new double[this.network.getLayer(i).getNode(j).getWeights().length + 1];
                System.arraycopy(this.network.getLayer(i).getNode(j).getWeights(), 0, network[i][j], 1, this.network.getLayer(i).getNode(j).getWeights().length);
                network[i][j][0] = this.network.getLayer(i).getNode(j).getBias();
            }
        }
        try {
            FileOutputStream fileOut = new FileOutputStream("neuralnet_0.009_3hidden.txt");
            ObjectOutputStream outNetwork = new ObjectOutputStream(fileOut);
            outNetwork.writeObject(network);

        } catch (IOException k) {
            System.out.println("error writing file");
        }
        return network;
    }
}
