#pragma once

#include "casadi/casadi.hpp"

using namespace casadi;
class Inversion {
  private:
    std::map<std::string, DM> arg_, res_;
    DM x0_  = DM(2, 1);
    DM p_   = DM(2, 1);
    DM lbx_ = DM(2, 1);
    DM ubx_ = DM(2, 1);
    Function csolver_;

    Dict solve_opts_;

    double last_converged_delta_ = 0;
    double last_converged_uy_ = 0;

  public:
    Inversion();
    void control(double* buffer);


};
