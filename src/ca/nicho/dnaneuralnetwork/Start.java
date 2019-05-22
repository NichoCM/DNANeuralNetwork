package ca.nicho.dnaneuralnetwork;

import java.util.ArrayList;

import ca.nicho.dnaneuralnetwork.screen.Screen;

public class Start {

	public static void main(String[] s) throws Exception {
			
		DNANeuralNetwork dnn = new DNANeuralNetwork(36, 8, 15, 1);
		for(int i = 0; i < 100; i++) {
			dnn.connectNodeRandom();
		}
				
		float[] input = new float[36];
		input[1] = 0.5f;
		dnn.printOutput(input);
		Screen screen = new Screen(dnn, 600, 600);
		
		/*ArrayList<DNANeuralNetwork> networks = new ArrayList<DNANeuralNetwork>();
				
		for(int i = 0; i < 2; i++) {
			networks.add(create());
		}
		
		networks.add(new DNANeuralNetwork(networks.get(0), networks.get(1)));
		
		int index = 0;
		Screen screen = new Screen(networks.get(index), 600, 600);
		while(index < networks.size()) {
			Thread.sleep(1000);
			screen.dnn = networks.get(index++);
			screen.repaint();
		}*/
		
	}
	
	public static DNANeuralNetwork create() {
		DNANeuralNetwork dnn = new DNANeuralNetwork(25, 8, 12, 2);
		for(int i = 0; i < 5; i++) {
			dnn.connectNodeRandom();
		}
		for(int i = 0; i < 2; i++) {
			dnn.mutateConnectionWeightRandom(0.2f);
		}
		return dnn;
	}
	
}
