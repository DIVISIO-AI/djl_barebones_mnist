package divisio.djl.example;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.Image.Flag;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import com.beust.jcommander.Parameter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * This is a heavily modified version of the DJL MNIST example found in the example project of DJL:
 * https://github.com/awslabs/djl/blob/master/examples/src/main/java/ai/djl/examples/training/TrainMnist.java
 * It has been modified to be instructive for Java devs which have no experience with Deep Learning and the DJL API and
 * to make some important aspects of the DJL API more visible.
 * We do not use the ready-made MNIST dataset of DJL on purpose - that way we see how to build a custom dataset and
 * load and prepare data for training.
 */
public class MnistTrainer extends JCommanderApplication{

    private static final Logger log = LoggerFactory.getLogger(MnistTrainer.class);

    @Parameter(names = {"-b", "--batch-size"},
            description = "Batch size to train with.")
    protected int batchSize = 256;
    @Parameter(names = {"-e", "--epochs"},
            description = "Number of epochs to train with.")
    protected int epochs = 3;
    @Parameter(names = {"-t", "--training-folder"},
            description = "Folder with training data.")
    protected String trainingFolderName = "mnist/train";
    @Parameter(names = {"-v", "--validation-folder"},
            description = "Folder with validation data.")
    protected String validationFolderName = "mnist/valid";
    @Parameter(names = {"-m", "--model-folder"},
            description = "Folder to save training progress (logs & models) to.")
    protected String modelFolder = this.getClass().getSimpleName();

    //the dataset for training
    private ImageClassifierDataset trainingSet;
    //the dataset for validating the training progress
    private ImageClassifierDataset validationSet;

    /**
     * Runs the training process including initialization, training loop and saving the model.
     */
    public void runTraining() throws IOException, TranslateException {
        //We log some info about the configuration we train with
        log.info("Training MNIST from folder '" +
                new File(trainingFolderName).getAbsolutePath() + "'.");
        log.info("Training will run for " +
                epochs + " Epochs, using batch size " + batchSize + ".");
        log.info("Results will be written to '" +
                new File(modelFolder).getAbsolutePath() + "'");
        //Make sure output folder exists
        new File(modelFolder).mkdirs();
        //Initialize data sets for training and validation
        this.trainingSet = buildDataset(new File(trainingFolderName));
        this.validationSet = buildDataset(new File(validationFolderName));
        //Build simple MNIST architecture:
        final Block neuralNet = new FullyConnectedMnistNetwork();
        //Now we create a model with our neural network.
        // A model is the combination of network structure and network parameters (learned weights).
        //A model can be trained and used for inference (=getting results/predictions).
        try (final Model model = Model.newInstance ("MNIST")) { // create an empty model
            //Set the network to be trained / used
            model.setBlock(neuralNet);
            //Create configuration for training (this creates logging, quality metrics etc.)
            //We use softmax cross entropy loss, as we do a multiclass-classification.
            final DefaultTrainingConfig trainingConfig =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                //We also evaluate accuracy, this is just for logging, but gives us better
                // feedback about the training progress.
                .addEvaluator(new Accuracy())
                //Add logging to the directory of the model file
                //(Here you can add custom listeners if you want more logging, e.g.
                // logging to a system like MLflow)
                .addTrainingListeners(
                        TrainingListener.Defaults.logging(new File(modelFolder).getAbsolutePath())
                )
                //Normally, there is no need to set the device, as the GPU is used automagically
                // if available.
                //If you want to compare performance, uncomment the following to use the CPU
                //.optDevices(new Device[]{ Device.cpu() })
                ;
            //To actually train the model,
            // we create a trainer with the training config we just built
            try (final Trainer trainer = model.newTrainer(trainingConfig)) {
                //This is a pitfall for DJL training: In order to get any feedback about training,
                // we have to set a new Metrics instance that will receive the training progress
                // from the listeners and metrics we just configured - if you omit this line,
                // everything will stay silent, nothing will be logged etc.
                // (this should be created by default, but isn't...)
                trainer.setMetrics(new Metrics());
                //Up until now, it still isn't clear what shape the input will actually have -
                // we need to initialize the training with the right input size. In our case we
                // load multiple images for each batch. (which each have three dimensions: width,
                // height and color channel). MNIST images are small, so width and height are both
                // 28 pixels, and as they are grayscale, the color dimension has size 1.
                //The batch size is usually the first dimension in deep learning frameworks.
                trainer.initialize(new Shape(batchSize, 28, 28, 1));
                //Now we can finally run the training process. We will train with the complete
                // training data as often as defined by the "epochs" parameter.
                // An "epoch" is one training with all training data available.
                for (int epochCount = 0; epochCount < epochs; ++epochCount) {
                    log.info("Starting epoch " + (epochCount + 1));
                    //run training for an epoch
                    trainEpoch(trainer);
                    //To make sure we do not loose our results, we save after each epoch.
                    // Not so important for MNIST, as training is very quick,
                    //but for really long trainings (which can take weeks) it is important to
                    // save your work regularly. (For huge training sets a single epoch may take
                    // days, in which case you should even save multiple times during an epoch.)
                    //We set the current epoch as metadata on the model.
                    model.setProperty("Epoch", String.valueOf(epochCount + 1));
                    //And then we save it to a file
                    // (folder and file name have to specified separately)
                    model.save(new File(modelFolder).toPath(), neuralNet.getClass().getSimpleName());
                }
            }
        }
    }

    protected ImageClassifierDataset buildDataset(final File folder) {
        return ImageClassifierDataset.builder()
                //Define folder to read from
                .setRootFolder(folder)
                //Set batch size & shuffle every time, this improves results.
                .setSampling(batchSize, true)
                //MNIST data contains no color information,
                // load as grayscale so we have only one color channel
                .optColorMode(Flag.GRAYSCALE)
                //Build a set with the given configuration
                .build();
    }

    /**
     * Performs one epoch of training with the given trainer.
     */
    protected void trainEpoch(final Trainer trainer) throws IOException, TranslateException {
        //In order to train, we go over all data in the dataset and perform a single training step.
        //The iterating is done by the trainer.
        // This is necessary so all GPU memory is properly cleaned up.
        for (final Batch batch : trainer.iterateDataset(trainingSet)) {
            try {
                //This is the actual Neural Network Training step.
                EasyTrain.trainBatch(trainer, batch);
                //Tell the trainer that the training step is over, this updates the network.
                trainer.step();
            } finally {
                //make sure batches are always closed, so we do not leak precious GPU memory
                batch.close();
            }
        }
        //We have used all the training data once - now check if this has improved things.
        // The following loop causes the evaluator we configured above to be used to test how the
        // current performance is.
        for (final Batch batch : trainer.iterateDataset(validationSet)) {
            try {
                EasyTrain.validateBatch(trainer, batch);
            } finally {
                batch.close();
            }
        }
        trainer.notifyListeners(listener -> listener.onEpoch(trainer));
    }

    public static void main(final String[] args){
        final MnistTrainer trainer = new MnistTrainer();
        trainer.parseArgs(args);
        try {
            trainer.runTraining();
        } catch (final Exception e) {
            System.err.println("Error during training.");
            e.printStackTrace();
        }
    }
}
