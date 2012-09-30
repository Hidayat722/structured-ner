import numpy as np

class StructuredPerceptron:

    w = None
    parameters_for_epoch = []

    def __init__(self, labels, feature_generator, epochs=10, eta=0.1):
        self.n_epochs = epochs
        self.labels = labels
        self.feature_generator = feature_generator
        self.n_features = feature_generator.n_features()
        self.eta = eta

    def train(self, sentences):

        self.w = np.zeros(self.n_features, dtype=float)

        for i_epoch in xrange(self.n_epochs):

            incorrect = 0
            total     = 0

            for sentence in sentences:
                total, incorrect = self.train_one(sentence, total, incorrect)

            self.parameters_for_epoch.append(self.w.copy())

            accuracy = 1.0 - (incorrect/float(total))

            print "Epoch %i, Accuracy: %f" % (i_epoch, accuracy)

        #Average!
        averaged_parameters = 0
        for epoch_parameters in self.parameters_for_epoch:
            averaged_parameters += epoch_parameters
        averaged_parameters /= float(len(self.parameters_for_epoch))

        self.w = averaged_parameters

        #Finished training
        self.trained = True


    def train_one(self, sentence, total, incorrect):

        z = self.viterbi_decode(sentence)

        for i in xrange(len(sentence.x)):

            total += 1

            y_i = sentence.y[i]
            z_i = z[i]

            if i == 0:
                y_prev = None
                z_prev = None
            else:
                y_prev = sentence.y[i-1]
                z_prev = z[i-1]

            if y_i != z_i:

                #The predicted tag was not the correct tag
                incorrect += 1

                correct_token_features = self.feature_generator.generate(sentence, i, y_prev, y_i)
                self.w[correct_token_features]   += self.eta

                predicted_token_features = self.feature_generator.generate(sentence, i, z_prev, z_i)
                self.w[predicted_token_features] -= self.eta

        return total, incorrect

    def f(self, sentence, i, last_label, current_label):
        f = np.zeros(self.n_features, dtype=float)
        f[self.feature_generator.generate(sentence, i, last_label, current_label)] = 1.
        return f

    def viterbi_decode(self, sentence):
        alphas = np.zeros((sentence.size(), len(self.labels)), dtype=float)
        path = {}

        for y in xrange(len(self.labels)):
            path[y] = [y]
            alphas[0, y] = sum(self.w * self.f(sentence, 0, None, y))

        for i in xrange(1, sentence.size()):
            new_path = {}

            for y in xrange(len(self.labels)):
                (alphas[i, y], b) = max([(sum(alphas[i-1, y_last] + self.w * self.f(sentence, i, y_last, y)), y_last) for y_last in xrange(len(self.labels))])
                new_path[y] = path[b] + [y]

            path = new_path

        return path[ alphas[:,-1].argmax() ]
