import theano
import numpy




def evaluate_fisher_info(func, input_img,predict,target, params):
    fisher_info = []
    for weight in params:
        fisher_info.append(numpy.zeros(weight.container.data.shape).astype(numpy.float32))

    n = input_img.shape[0]
    param_len = len(fisher_info)

    for i in range(0,n):
        print("Executing sample {}".format(i) )
        #print(predict[i])
        eps = 0.000001
        s_cat = numpy.random.multinomial(1,predict[i]/(numpy.sum(predict[i])+eps),size = 1)
        #print(numpy.argmax(s_cat) == target[i])
        grad_set = func(input_img[i].reshape(1 , 784), numpy.argmax(s_cat).reshape(1,).astype(numpy.int32))
        #ind = numpy.random.randint(0,10,1)
        #grad_set = func(input_img[i].reshape(1, 784), ind.reshape(1, ).astype(numpy.int32))
        for j in range(0,param_len):
            cpu_grad = numpy.asarray(grad_set[j])
            #print(numpy.max(cpu_grad))
            #fisher_info[j] = 1.0*i/(i+1)*fisher_info[j] + 1.0/(i+1)*cpu_grad*cpu_grad
            fisher_info[j] += numpy.square(cpu_grad)/n


    return fisher_info

