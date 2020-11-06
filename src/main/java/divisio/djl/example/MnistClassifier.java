package divisio.djl.example;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.nn.Block;
import ai.djl.translate.TranslateException;
import com.beust.jcommander.Parameter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A simple application that shows how to run classification based on a previous training result.
 * Adopted and modified from:
 * https://github.com/awslabs/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/ImageClassification.java
 */
public class MnistClassifier extends JCommanderApplication {

    private static final Logger log = LoggerFactory.getLogger(MnistClassifier.class);

    @Parameter(names = {"-m", "--model-folder"}, description = "Folder to load trained model from.")
    protected String modelFolder = MnistTrainer.class.getSimpleName();

    @Parameter(description = "Image files to classify, a list of files and/or folders.")
    protected List<String> imageFileNames = new ArrayList<>();

    /**
     * Utility function to list all files defined by the {@code imageFileNames} property.
     * If an entry is a file, only that is returned, if it is a folder,
     * all files in that folder are returned recursively.
     */
    private List<Path> listAllFiles() {
        return imageFileNames.stream().flatMap((name) -> {
            try {
                return Files.walk(Paths.get(name));
            } catch (IOException e) {
                return Stream.empty();
            }
        }).filter(Files::isRegularFile).collect(Collectors.toList());
    }

    /**
     * Initializes the model, loads the trained model parameters and runs classification on all
     * image files provided.
     */
    public void runClassification()
            throws IOException, MalformedModelException, TranslateException {
        //Create the same network architecture as the trainer.
        final Block neuralNet = new FullyConnectedMnistNetwork();
        //As with training, create an empty model with the architecture.
        try (final Model model = Model.newInstance("MNIST")) {
            //Tell the model to use the MNIST classifier network.
            model.setBlock(neuralNet);
            //Now, instead of training, we load the trained parameters for using them in classification.
            model.load(new File(modelFolder).toPath(), neuralNet.getClass().getSimpleName());
            //Translators turn input/output for neural networks (which are always NDLists) from/to other classes that
            //are the actual input/output. In our case, the input is an image and the output is a classification result.
            final MnistClassificationTranslator translator = new MnistClassificationTranslator();
            //We use the translator to create a predictor from the model and the translator. The Predictor is the
            //sibling of the Trainer. It serves the same purpose for inference as the Trainer for training.
            //It does all memory management and runs the actual prediction process on the neural network in the model.
            try (final Predictor<File, Classifications> predictor = model.newPredictor(translator)) {
                //We loop over all files given on the command line or contained in folders given on the command line
                for (final Path imageFile : listAllFiles()) {
                    //log the file name so we now what we are classifying
                    log.info(imageFile.toString());
                    //This is the actual prediction step - the translator allows the predictor to return an actual
                    //result we can use instead of an NDArray that needs further interpretation.
                    final Classifications predictionResult = predictor.predict(imageFile.toFile());
                    log.info(predictionResult.toString());
                }
            }
        }
    }



    public static void main(final String[] args) {
        final MnistClassifier classifier = new MnistClassifier();
        classifier.parseArgs(args);
        try {
            if (classifier.imageFileNames.isEmpty()) {
                System.err.println("You did not specify any images to classify - " +
                        "you can find test images in the mnist/test folder.");
                System.exit(-1);
                return;
            }
            classifier.runClassification();
        } catch (final Exception e) {
            System.err.println("Error during classification.");
            e.printStackTrace();
        }
    }
}
