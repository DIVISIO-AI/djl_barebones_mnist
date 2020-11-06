package divisio.djl.example;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;

/**
 * Base class for command line applications that takes care of command line parsing and printing help messages.
 */
public class JCommanderApplication {

    @Parameter(names = {"-h", "--help"}, description = "Display command line usage.", help=true)
    protected boolean help = false;

    /**
     * Parses command line arguments, if that fails, exits the current process. Prints help messages if requested.
     * @param args the args passed to the main method.
     */
    public void parseArgs(final String[] args) {
        //Parse command line params
        final JCommander commander = JCommander.newBuilder().addObject(this).build();
        try {
            commander.parse(args);
        } catch (final ParameterException pe) {
            //Thrown if the given arguments are invalid - print the error message, print usage instructions & exit.
            System.out.println(pe.getMessage());
            commander.usage();
            System.exit(-1);
            return;
        }
        if (this.help) {
            commander.usage();
            System.exit(0);
            return;
        }
    }
}
