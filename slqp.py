# *******************************************************
# * Copyright (c) 2020 by Artelys                       *
# * All Rights Reserved                                 *
# *******************************************************
from numpy.linalg import multi_dot
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This example demonstrates how to use Knitro to solve the following
# mixed-integer nonlinear optimization problem (MINLP).  This model
# is test problem 1 (Synthesis of processing system) in
# M. Duran & I.E. Grossmann,  "An outer approximation algorithm for
# a class of mixed integer nonlinear programs", Mathematical
# Programming 36, pp. 307-339, 1986.  The problem also appears as
# problem synthes1 in the MacMINLP test set.
#
# min   5 x3 + 6 x4 + 8 x5 + 10 x0 - 7 x2 -18 log(x1 + 1)
#      - 19.2 log(x0 - x1 + 1) + 10
# s.t.  0.8 log(x1 + 1) + 0.96 log(x0 - x1 + 1) - 0.8 x2 >= 0
#       log(x1 + 1) + 1.2 log(x0 - x1 + 1) - x2 - 2 x5 >= -2
#       x1 - x0 <= 0
#       x1 - 2 x3 <= 0
#       x0 - x1 - 2 x4 <= 0
#       x3 + x4 <= 1
#       0 <= x0 <= 2
#       0 <= x1 <= 2
#       0 <= x2 <= 1
#       x0, x1, x2 continuous
#       x3, x4, x5 binary
#
# The solution is (1.30098, 0, 1, 0, 1, 0).


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

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from knitro import *
import math
# *------------------------------------------------------------------*
# *     GLOBALS                               *
# *------------------------------------------------------------------*
gamma_c = 1
gamma_I = 1
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
    evalResult.obj = ((Sz * Sz) * gamma_c ) + gamma_I * ((Ia/Ic)*(Ia/Ic) + (Ib/Ic)*(Ib/Ic)) + gamma_L * 0.5 * multi_dot((x, laplacian,x))
    # Evaluate nonlinear constraint structure
    evalResult.c[0] = s_omega_omega_tag(1)
    evalResult.c[1] = s_omega_omega_tag(2)
    evalResult.c[2] = s_omega_omega_tag(4)
    evalResult.c[3] = s_omega_omega_tag(5)
    evalResult.c[4] = s_omega_omega_tag(6)
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

#   5 x3 + 6 x4 + 8 x5 + 10 x0 - 7 x2 -18 log(x1 + 1)
#      - 19.2 log(x0 - x1 + 1) + 10
    # Evaluate gradient of nonlinear objective structure
    dTmp1 = x[0] - x[1] + 1.0
    dTmp2 = x[1] + 1.0
    evalResult.objGrad[0] = -(19.2 / dTmp1)
    evalResult.objGrad[1] = (-18.0 / dTmp2) + (19.2 / dTmp1)

    # Gradient of nonlinear structure in constraint 0.
    evalResult.jac[0] = 0.96 / dTmp1  # wrt x0
    evalResult.jac[1] = (-0.96 / dTmp1) + (0.8 / dTmp2)  # wrt x1
    # Gradient of nonlinear structure in constraint 1.
    evalResult.jac[2] = 1.2 / dTmp1  # wrt x0
    evalResult.jac[3] = (-1.2 / dTmp1) + (1.0 / dTmp2)  # wrt x1

    return 0


# *------------------------------------------------------------------*
# *     FUNCTION callbackEvalH                                       *
# *------------------------------------------------------------------*
# The signature of this function matches KN_eval_callback in knitro.py.
# Only "hess" or "hessVec" are set in the KN_eval_result structure.
def callbackEvalH(kc, cb, evalRequest, evalResult, userParams):
    if evalRequest.type != KN_RC_EVALH and evalRequest.type != KN_RC_EVALH_NO_F:
        print("*** callbackEvalH incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x
    lambda_ = evalRequest.lambda_
    # Scale objective component of hessian by sigma
    sigma = evalRequest.sigma

    # Evaluate the non-zero components in the Hessian of the Lagrangian.
    # Note: Since the Hessian is symmetric, we only provide the
    #       nonzero elements in the upper triangle (plus diagonal).
    dTmp1 = x[0] - x[1] + 1.0
    dTmp2 = x[1] + 1.0
    evalResult.hess[0] = sigma * (19.2 / (dTmp1 * dTmp1))  # (0,0)
    + lambda_[0] * (-0.96 / (dTmp1 * dTmp1))
    + lambda_[1] * (-1.2 / (dTmp1 * dTmp1))
    evalResult.hess[1] = sigma * (-19.2 / (dTmp1 * dTmp1))  # (0,1)
    + lambda_[0] * (0.96 / (dTmp1 * dTmp1))
    + lambda_[1] * (1.2 / (dTmp1 * dTmp1))
    evalResult.hess[2] = sigma * ((19.2 / (dTmp1 * dTmp1)) + (18.0 / (dTmp2 * dTmp2)))  # (1,1)
    + lambda_[0] * ((-0.96 / (dTmp1 * dTmp1)) - (0.8 / (dTmp2 * dTmp2)))
    + lambda_[1] * ((-1.2 / (dTmp1 * dTmp1)) - (1.0 / (dTmp2 * dTmp2)))

    return 0


# *------------------------------------------------------------------*
# *     FUNCTION callbackProcessNode                                 *
# *------------------------------------------------------------------*
# The signature of this function matches KN_user_callback in knitro.py.
# Argument "kcSub" is the context pointer for the last node
# subproblem solved inside Knitro.  The application level context
# pointer is passed in through "userParams".
def callbackProcessNode(kcSub, x, lambda_, userParams):
    # The Knitro context pointer was passed in through "userParams".
    kc = userParams

    # Print info about the status of the MIP solution.
    numNodes = KN_get_mip_number_nodes(kc)
    relaxBound = KN_get_mip_relaxation_bnd(kc)
    # Note: To retrieve solution information about the node subproblem
    # we need to pass in "kcSub" here.
    nodeObj = KN_get_obj_value(kcSub)
    print("callbackProcessNode:")
    print("    Node number    = %d" % numNodes)
    print("    Node objective = %e" % nodeObj)
    print("    Current relaxation bound = %e" % relaxBound)
    try:
        print("    Current incumbent bound  = %e" % KN_get_mip_incumbent_obj(kc))
        print("    Absolute integrality gap = %e" % KN_get_mip_abs_gap(kc))
        print("    Relative integrality gap = %e" % KN_get_mip_rel_gap(kc))
    except:
        print("    No integer feasible point found yet.")

    # User defined termination example.
    # Uncomment below to force termination after 3 nodes.
    # if (numNodes == 3)
    #    return KN_RC_USER_TERMINATION

    return 0


# *------------------------------------------------------------------*
# *     main                                                         *
# *------------------------------------------------------------------*
def slqp(laplacian_given , s_omega_given, s_omega_k_given):
    """

    :param laplacian_given:
    :param s_omega_given:
    :param s_omega_k_given:
    :return:
    """
    global k, s_omega, s_omega_k, laplacian
    laplacian, s_omega = laplacian_given, s_omega_given
    k = s_omega_k_given.shape[0]
    s_omega_k = s_omega_k_given

    # Create a new Knitro solver instance.
    try:
        kc = KN_new()
    except:
        print("Failed to find a valid license.")
        quit()

    # Illustrate how to override default options.
    KN_set_int_param(kc, "algorithm", KN_ALG_ACT_CG)

    # Initialize Knitro with the problem definition.

    # Add the variables and set their bounds and types.
    # Note: any unset lower bounds are assumed to be
    # unbounded below and any unset upper bounds are
    # assumed to be unbounded above.
    KN_add_vars(kc, k)
    KN_set_var_lobnds(kc, xLoBnds=[0] * k)
    KN_set_var_upbnds(kc, xUpBnds=[1] * k)

    # Add the constraints and set their bounds
    KN_add_cons(kc, 5)
    KN_set_con_eqbnds (kc, cEqBnds = [0.0, 0.0, 0.0, 0.0, 0.0]);

    # Add a callback function "callbackEvalFC" to evaluate the nonlinear
    # structure in the objective and first two constraints.  Note that
    # the linear terms in the objective and first two constraints were
    # added above in "KN_add_obj_linear_struct()" and
    # "KN_add_con_linear_struct()" and will not be specified in the
    # callback.
    cIndices = [0, 1, 2, 3, 4]  # Constraint indices for callback
    cb = KN_add_eval_callback(kc, evalObj=True, indexCons=cIndices, funcCallback=callbackEvalFC)

    # Also add a callback function "callbackEvalGA" to evaluate the
    # gradients of all nonlinear terms specified in the callback.  If
    # not provided, Knitro will approximate the gradients using finite-
    # differencing.  However, we recommend providing callbacks to
    # evaluate the exact gradients whenever possible as this can
    # drastically improve the performance of Knitro.
    # Objective gradient non-zero structure for callback
    # objGradIndexVarsCB = [0, 1]
    # Constraint Jacobian non-zero structure for callback
    # jacIndexConsCB = [0, 0, 1, 1]
    # jacIndexVarsCB = [0, 1, 0, 1]
    # KN_set_cb_grad(kc, cb, objGradIndexVars=objGradIndexVarsCB, jacIndexCons=jacIndexConsCB, jacIndexVars=jacIndexVarsCB,
    #                gradCallback=callbackEvalGA)
    #
    # hessIndexVars1CB = [0, 0, 1]
    # hessIndexVars2CB = [0, 1, 1]
    # Add a callback function "callbackEvalH" to evaluate the Hessian
    # (i.e. second derivative matrix) of the objective.  If not specified,
    # Knitro will approximate the Hessian. However, providing a callback
    # for the exact Hessian (as well as the non-zero sparsity structure)
    # can greatly improve Knitro performance and is recommended if possible.
    # Since the Hessian is symmetric, only the upper triangle is provided.
    # KN_set_cb_hess(kc, cb, hessIndexVars1=hessIndexVars1CB, hessIndexVars2=hessIndexVars2CB, hessCallback=callbackEvalH)

    # Specify that the user is able to provide evaluations
    #  of the Hessian matrix without the objective component.
    #  turned off by default but should be enabled if possible.
    # KN_set_int_param(kc, KN_PARAM_HESSIAN_NO_F, KN_HESSIAN_NO_F_ALLOW)

    # Set minimize or maximize (if not set, assumed minimize)
    KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE)

    # Set a callback function that performs some user-defined task
    # after completion of each node in the branch-and-bound tree.
    # KN_set_mip_node_callback(kc, callbackProcessNode, kc)

    # Solve the problem.
    #
    # Return status codes are defined in "knitro.py" and described
    # in the Knitro manual.

    nStatus = KN_solve(kc)
    # An example of obtaining solution information.
    nSTatus, objSol, x, lambda_ = KN_get_solution(kc)

    # print("Optimal objective value  = %e", objSol)
    # print("Optimal x")
    # for i in range(k):
    #     print("  x[%d] = %e" % (i, x[i]))

    # Delete the Knitro solver instance.
    KN_free(kc)
    return x