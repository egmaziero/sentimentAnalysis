package ufla.br.sentimentAnalysis;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import java.util.Properties;

import org.apache.log4j.BasicConfigurator;

public class App {
	public static void main(String[] args) {
		BasicConfigurator.configure();

		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, lemma, parse, sentiment");

		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

		String text = "This software is awsome! It wordked very well with my smartphone since the first release!";

		// create an empty Annotation just with the given text
		Annotation document = new Annotation(text);

		// run all Annotators on this text
		pipeline.annotate(document);

		for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
			System.out.println("---");
			System.out.println(sentence);
			// System.out.println(sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class));
			Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
			int sentiment = RNNCoreAnnotations.getPredictedClass(tree);
			System.out.println(sentence.get(SentimentCoreAnnotations.SentimentClass.class));
			System.out.println(sentiment);
		}
	}
}
