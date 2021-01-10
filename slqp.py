# *******************************************************
# * Copyright (c) 2020 by Artelys                       *
# * All Rights Reserved                                 *
# *******************************************************
import numpy as np
from knitro import *
import math

# s_omega_omega_tag = s_everything - np.dot(beta, s_omega_k_matrix) # len 10, function of beta

# min gamma_c l^2 M^2 + gamma_I ( (Ia/Ic)^2 + (Ib/Ic)^2) + gamma_L 0.5 [beta]T L [betta]

#  CONSTRAINTS:
# s.t. s_omega_omega_tag[1] = 0
# s_omega_omega_tag[2] = 0
# s_omega_omega_tag[4] =0  (cos(phi)sin(phi)(sx^2 -sy^2)+(cos(phi)^2-sin(phi)^2)Sxy =0 , phi=0)
# s_omega_omega_tag[5] = 0
# s_omega_omega_tag[6] = 0
#  0<= beta[i] <=1 for all i
# s_omega_omega_tag[3] = l*M (redundant)
# beta continuous

# express with beta:
# gamma_c, gamma_I, gamma_L
# l => s_omega_omega_tag[3]/s_omega_omega_tag[0] (sz/M)
# # M => s_omega_omega_tag[0]
# # Ia => s_omega_omega_tag[8]+ s_omega_omega_tag[9] (sy2+sz2)
# # Ib => s_omega_omega_tag[7]+ s_omega_omega_tag[9] (sx2+sz2)
# # Ic => s_omega_omega_tag[7]+ s_omega_omega_tag[8] (sx2+sy2)

# *------------------------------------------------------------------*
# *     GLOBALS                                                      *
# *------------------------------------------------------------------*
gamma_c = 1
gamma_I = 100
gamma_L = 1
laplacian = None
k = None
s_omega_k = None
s_omega = None

# *------------------------------------------------------------------*
# *     FUNCTION callbackEvalFC                                      *
# *------------------------------------------------------------------*
# The signature of this function matches KN_eval_callback in knitro.py.
# Only "obj" and "c" are set in the KN_eval_result structure.
def callbackEvalFC(kc, cb, evalRequest, evalResult, userParams):
    global gamma_c ,gamma_I ,gamma_L, laplacian
    if evalRequest.type != KN_RC_EVALFC:
        print("*** callbackEvalFCOurs incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x

    # s_everything - np.dot(beta, s_omega_k_matrix)[3] / s_everything - np.dot(beta, s_omega_k_matrix)[0]
    # s_everything - np.dot(beta, s_omega_k_matrix)

    # Evaluate nonlinear objective structure
    teta = 0
    def s_omega_omega_tag(index):
        global s_omega_k, s_omega
        sum =0
        for i in range(len(x)):
            sum += s_omega_k[i][index]*x[i]
        return s_omega[index] - sum
    Sz = s_omega_omega_tag(3)
    Sx2 = s_omega_omega_tag(7)
    Sy2 = s_omega_omega_tag(8)
    Sz2 = s_omega_omega_tag(9)
    Ia = Sy2 + Sz2
    Ib = Sx2 + Sz2
    Ic = Sx2+ Sy2
    evalResult.obj = ((Sz * Sz) * gamma_c ) + gamma_I * ((Ia/Ic)*(Ia/Ic) + (Ib/Ic)*(Ib/Ic)) + gamma_L * 0.5 * np.dot(np.dot(x, laplacian), x)
    # Evaluate nonlinear constraint structure
    evalResult.c[0] = s_omega_omega_tag(1)
    evalResult.c[1] = s_omega_omega_tag(2)
    evalResult.c[3] = s_omega_omega_tag(5)
    evalResult.c[4] = s_omega_omega_tag(6)
    evalResult.c[2] = (s_omega_omega_tag(7)- s_omega_omega_tag(8))*np.cos(teta)*np.sin(teta) + (np.cos(teta)*np.cos(teta)-np.sin(teta)*np.sin(teta))* s_omega_omega_tag(4)
    return 0


# *------------------------------------------------------------------*
# *     FUNCTION callbackEvalGA                                      *
# *------------------------------------------------------------------*
# The signature of this function matches KN_eval_callback in knitro.py.
# Only "objGrad" and "jac" are set in the KN_eval_result structure.
def callbackEvalGA(kc, cb, evalRequest, evalResult, userParams):
    if evalRequest.type != KN_RC_EVALGA:
        print("*** callbackEvalGA incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x

    # Evaluate gradient of nonlinear objective structure
    global s_omega_k

    def s_omega_omega_tag(index):
        global s_omega_k, s_omega
        sum = 0
        for i in range(len(x)):
            sum += s_omega_k[i][index] * x[i]
        return s_omega[index] - sum

    Sz = s_omega_omega_tag(3)
    Sx2 = s_omega_omega_tag(7)
    Sy2 = s_omega_omega_tag(8)
    Sz2 = s_omega_omega_tag(9)
    Ia = Sy2 + Sz2
    Ib = Sx2 + Sz2
    Ic = Sx2 + Sy2

    #    evalResult.obj = ((Sz * Sz) * gamma_c ) + gamma_I * ((Ia/Ic)*(Ia/Ic) + (Ib/Ic)*(Ib/Ic)) + gamma_L * 0.5 * multi_dot((x, laplacian,x))
    for i in range(len(x)):
        evalResult.objGrad[i] = 2 * Sz * -s_omega_k[i][3] * gamma_c
        evalResult.objGrad[i] += gamma_I * 2 * (Ia/Ic) * ((Ic *(-s_omega_k[i][8]-s_omega_k[i][9]) - Ia * (-s_omega_k[i][8]-s_omega_k[i][7]))/(Ic**2))
        evalResult.objGrad[i] += gamma_I * 2 * (Ib/Ic) * ((Ic *(-s_omega_k[i][7]-s_omega_k[i][9]) - Ib * (-s_omega_k[i][8]-s_omega_k[i][7]))/(Ic**2))
        evalResult.objGrad[i] +=  gamma_L * 0.5 * np.dot((laplacian[i] + laplacian[:,i]), x)


    # Gradient of nonlinear structure in constraint 0.
    i = 0
    for s_num in [1 ,2 ,4 ,5 ,6]:
        for j in range(len(x)):
            evalResult.jac[i] = -s_omega_k[j][s_num]
            i += 1
    return 0

# *------------------------------------------------------------------*
# *     Main                                                         *
# *------------------------------------------------------------------*
def slqp(laplacian_given , s_omega_given, s_omega_k_given):
    """
    main function of the solver
    :param laplacian_given:
    :param s_omega_given:
    :param s_omega_k_given:
    :return:
    """
    global k, s_omega, s_omega_k, laplacian
    scale = 0.001
    laplacian, s_omega = laplacian_given, s_omega_given*scale
    k = s_omega_k_given.shape[0]
    s_omega_k = s_omega_k_given*scale

    # Create a new Knitro solver instance.
    try:
        kc = KN_new()
    except:
        print("Failed to find a valid license.")
        quit()

    # Illustrate how to override default options.
    KN_set_int_param(kc, "algorithm", KN_ALG_ACT_CG)
    # KN_set_int_param(kc, KN_PARAM_MULTISTART, KN_MULTISTART_YES)
    #
    # Initialize Knitro with the problem definition.

    # Add the variables and set their bounds and types.
    KN_add_vars(kc, k)
    KN_set_var_lobnds(kc, xLoBnds=[0] * k)
    KN_set_var_upbnds(kc, xUpBnds=[1] * k)

    # Define an initial point.  If not set, Knitro will generate one.
    KN_set_var_primal_init_values(kc, xInitVals=[0] * k)

    # Add the constraints and set their bounds
    KN_add_cons(kc, 5)
    KN_set_con_lobnds(kc, cLoBnds=np.array([-10.0]*5))
    KN_set_con_upbnds(kc, cUpBnds=np.array([10.0]*5))
    # KN_set_con_eqbnds (kc, cEqBnds = [0.0, 0.0, 0.0, 0.0, 0.0]);

    # Add a callback function "callbackEvalFC" to evaluate the nonlinear
    # structure in the objective and constraints.
    cIndices = [0, 1, 2, 3, 4]  # Constraint indices for callback
    cb = KN_add_eval_callback(kc, evalObj=True,
                              indexCons=cIndices,
                              funcCallback=callbackEvalFC)

    # Also add a callback function "callbackEvalGA" to evaluate the
    # gradients of all nonlinear terms specified in the callback.  If
    # not provided, Knitro will approximate the gradients using finite-
    # differencing.
    objGradIndexVarsCB = [x for x in range(k)]
    # Constraint Jacobian non-zero structure for callback
    jacIndexConsCB = [i for i in range(5) for _ in range(k)]
    jacIndexVarsCB = [i for _ in range(5) for i in range(k)]
    # KN_set_cb_grad(kc, cb, objGradIndexVars=objGradIndexVarsCB, jacIndexCons=jacIndexConsCB, jacIndexVars=jacIndexVarsCB,
    #                gradCallback=callbackEvalGA)

    # Approximate hessian using BFGS
    KN_set_int_param(kc, "hessopt", KN_HESSOPT_LBFGS)

    # Set minimize or maximize (if not set, assumed minimize)
    KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE)

    # Solve the problem.
    nStatus = KN_solve(kc)
    nSTatus, objSol, x, lambda_ = KN_get_solution(kc)

    # Delete the Knitro solver instance.
    KN_free(kc)
    print(x)
    return x