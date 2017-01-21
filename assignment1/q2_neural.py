import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    N = data.shape[0]
    
     
    h = sigmoid(data.dot(W1) + b1)
    prob = softmax(h.dot(W2) + b2)
    
    
                                                                  
#    print labels
#    print np.log(prob)
#    print labels * np.log(prob)
    cost = -np.sum(labels * np.log(prob))
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    #print prob.shape
    #print labels.shape
    gradtheta2 = prob - labels    # N * Dy
    
    gradW2 = (h.T).dot(gradtheta2)  # H * N   N* Dy   H * Dy
    #print gradW2.shape
    gradb2 = np.sum(gradtheta2, 0) #pass
    #print gradb2.shape
    gradh  = gradtheta2.dot(W2.T)   
    
    gradtheta1 = gradh * sigmoid_grad(h)
    #gradtheta1 = gradh * (1 - gradh)
    gradW1 = (data.T).dot(gradtheta1) 
    gradb1 = np.sum(gradtheta1, 0)

    ### END YOUR  CODE
    
    ### Stack gradients (do not modify)
    #print gradW1.shape
    #print gradb1.shape
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    #print grad.shape
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()