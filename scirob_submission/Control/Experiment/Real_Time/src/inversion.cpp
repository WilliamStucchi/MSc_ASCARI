#include "inversion.h"

Inversion::Inversion() {
  solve_opts_["print_time"]                    = false;
  solve_opts_["ipopt.linear_solver"]           = "ma27";
  solve_opts_["ipopt.max_iter"]                = 60;
  //solve_opts_["ipopt.max_cpu_time"]            = 0.099;
  solve_opts_["ipopt.print_level"]             = 0;
  //solve_opts_["ipopt.tol"]                     = .01;
  //solve_opts_["ipopt.constr_viol_tol"]         = .01;
  //solve_opts_["ipopt.compl_inf_tol"]           = .01;
  //solve_opts_["ipopt.dual_inf_tol"]            = 1e-4; // mb: I believe this is stationarity
  //solve_opts_["ipopt.hessian_approximation"]   = "limited-memory";
  solve_opts_["ipopt.warm_start_init_point"]   = "no";

  csolver_ = nlpsol("csolver", "ipopt", "libcasadi_solver.so", solve_opts_);

}


void Inversion::control(double* buffer) {

  double ux = buffer[0];
  double k  = buffer[1];


  x0_(0) = k*2.6;   // guess for delta
  x0_(1) = 0;       // guess for uy
  arg_["x0"]   = x0_;

  p_(0) = ux;       // 
  p_(1) = k;        // 
  arg_["p"]    = p_;

  lbx_(0) = -25*3.14/180.0;
  ubx_(0) =  25*3.14/180.0;
  lbx_(1) = -10;
  ubx_(1) =  10;
  arg_["lbx"]  = lbx_;
  arg_["ubx"]  = ubx_;

  res_ = csolver_(arg_);
  std::vector<double> ipopt_x(res_.at("x"));

  Dict solve_stats = csolver_.stats();
  int iterations = solve_stats["iter_count"].to_int();
  double solve_time = solve_stats["t_wall_csolver"].to_double();
  std::string status = solve_stats["return_status"].to_string();
  Dict iteration_stats = solve_stats["iterations"];
  std::vector<double> obj = iteration_stats["obj"].to_double_vector();
  double objective = obj.back();
  int exit_flag;
  // vincent's convention
  if      (status.compare("Solve_Succeeded"                   )==0) { exit_flag =    1; }
  else if (status.compare("Solved_To_Acceptable_Level"        )==0) { exit_flag =    2; }
  else if (status.compare("User_Requested_Stop"               )==0) { exit_flag =    3; }
  else if (status.compare("Feasible_Point_Found"              )==0) { exit_flag =    4; }
  else if (status.compare("Maximum_Iterations_Exceeded"       )==0) { exit_flag =   -1; }
  else if (status.compare("Restoration_Failed"                )==0) { exit_flag =   -2; }
  else if (status.compare("Error_In_Step_Computation"         )==0) { exit_flag =   -3; }
  else if (status.compare("Maximum_CpuTime_Exceeded"          )==0) { exit_flag =   -4; }
  else if (status.compare("Infeasible_Problem_Detected"       )==0) { exit_flag =   -5; }
  else if (status.compare("Search_Direction_Becomes_Too_Small")==0) { exit_flag =   -6; }
  else if (status.compare("Diverging_Iterates"                )==0) { exit_flag =   -7; }
  else if (status.compare("Not_Enough_Degrees_Of_Freedom"     )==0) { exit_flag =  -10; }
  else if (status.compare("Invalid_Problem_Definition"        )==0) { exit_flag =  -11; }
  else if (status.compare("Invalid_Option"                    )==0) { exit_flag =  -12; }
  else if (status.compare("Unrecoverable_Exception"           )==0) { exit_flag = -100; }
  else if (status.compare("NonIpopt_Exception_Thrown"         )==0) { exit_flag = -101; }
  else if (status.compare("Insufficient_Memory"               )==0) { exit_flag = -102; }
  else if (status.compare("Internal_Error"                    )==0) { exit_flag = -199; }
  else                                                              { exit_flag = -999; }


  iterations = -iterations;
  if ((exit_flag==1 or exit_flag==2) and objective < 1e-3) {
    last_converged_delta_   = ipopt_x[0];
    last_converged_uy_      = ipopt_x[1];
    iterations = -iterations;

  }

  std::cout << status << ": delta=" << ipopt_x[0]*180/3.14 << "[deg], uy=" << ipopt_x[1] <<
   ", obj=" << objective << " in " << solve_time*1000 << " ms and " << iterations << " iter" << std::endl;


  buffer[0] = last_converged_delta_;
  if (ux < 1){ux=1;}
  buffer[1] = std::atan(last_converged_uy_/ux);
  buffer[2] = iterations;
  buffer[3] = solve_time;
  buffer[4] = objective;

}
