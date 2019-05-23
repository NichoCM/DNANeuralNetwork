package ca.nicho.dnaneuralnetwork.screen;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Stroke;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.util.HashMap;

import javax.swing.JFrame;
import javax.swing.JPanel;

import ca.nicho.dnaneuralnetwork.DNANeuralNetwork;

public class Screen extends JPanel {

	public static final int MARGIN = 60;
	public static final int NODE_SIZE = 15;
	
	public static final Stroke LINE_STROKE = new BasicStroke(3);
	public static final Font TEXT_FONT = new Font("Calibri", Font.PLAIN, 18);
	
	public DNANeuralNetwork dnn;
	
	public int generation = 0;
	public int index = 0;
	
	public int width;
	public int height;
	
	private HashMap<Integer, Point> nodeCache;
	
	public Screen(int width, int height) {		
		// Create the JFrame for this panel
		this.width = width;
		this.height = height;
		
		JFrame frame = new JFrame();
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setPreferredSize(new Dimension(width, height));
		frame.setContentPane(this);
		frame.pack();
		
		// Handle frame resize event
		frame.getRootPane().addComponentListener(new ComponentAdapter() {
            public void componentResized(ComponentEvent e) {
                Screen.this.width = Screen.this.getWidth();
                Screen.this.height = Screen.this.getHeight();
            }
        });
			
	}
	
	public void setNetwork(DNANeuralNetwork dnn) {
		this.dnn = dnn;
		this.repaint();
	}
	
	@Override
	public void paintComponent(Graphics g3d) {
		super.paintComponent(g3d);
		
		if(this.dnn == null) return;
		
		Graphics2D g = (Graphics2D)g3d;
		
		// Reset the node cache
		nodeCache = new HashMap<Integer, Point>();
		
		int layer = 0;
		
		// Draw input layers
		for(int i = 0; i < dnn.inputSize; i++) {
			drawNode(g, i, dnn.inputSize, layer);		
		}
		layer++;
		
		// Draw middle layers
		for(int i = 0; i < dnn.depth; i++) {
			for(int o = 0; o < dnn.middleSize; o++) {
				drawNode(g, o, dnn.middleSize, layer);
			}
			layer ++;
		}
		
		// Draw output layer
		for(int i = 0; i < dnn.inputSize; i++) {
			drawNode(g, i, dnn.outputSize, layer);
		}
		
		// Draw connections
		for(int i = 0; i < dnn.size; i++) {
			for(int j = 0; j < dnn.size; j++) {
				if(dnn.graph[i][j] != 0) {	
					drawConnection(g, i, j);
				}
			}
		}
		
		g.setColor(Color.black);
		g.setFont(TEXT_FONT);
		g.drawString("Genetic Hash: " + dnn.toString(), 20, 20);
		g.drawString("Generation: " + generation, 20, 35);
		g.drawString("Current index: " + index, 20, 50);
		
	}
	
	private void drawNode(Graphics2D g, int index, int nodeCount, int layer) { 
		
		float value = dnn.cachedActivtations[index];
		
		int paddingLeft = (int)(width / (float)(dnn.depth + 2) - NODE_SIZE / 2) / 2;
		int paddingTop = (int)(height / (float)nodeCount - NODE_SIZE / 2) / 2;
		
		int left = (int)((layer / (float)(dnn.depth + 2)) * (width - 2 * MARGIN)) + MARGIN;
		int top = (int)((index / (float)nodeCount) * (height - 2 * MARGIN)) + MARGIN;
		
		Point p = new Point(left + paddingLeft, top + paddingTop);
		nodeCache.put(nodeCache.size(), p);
		
		if(value < 0) {
			g.setColor(Color.magenta);
		} else if(value > 0) {
			g.setColor(Color.orange);
		} else {
			g.setColor(Color.black);
		}
		g.fillOval(p.x, p.y, NODE_SIZE, NODE_SIZE);
	}
	
	private void drawConnection(Graphics2D g, int i, int j) {
		
		Point p1 = nodeCache.get(i);
		Point p2 = nodeCache.get(j);
		
		float value = this.dnn.graph[i][j];
		
		g.setColor(new Color(0f, value < 0 ? 1f : 0f, value > 0 ? 1f : 0f, Math.abs(value)));
		g.setStroke(LINE_STROKE);
		g.drawLine(p1.x + NODE_SIZE / 2, p1.y + NODE_SIZE / 2, p2.x + NODE_SIZE / 2, p2.y + NODE_SIZE / 2);
		
	}
	
}
