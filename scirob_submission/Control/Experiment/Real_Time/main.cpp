


#include "inversion.h"
#include "udp.h"



int main() {

  // udp, inverter objects
  UDP udp(4545, 4545);
  Inversion inversion;

  // buffer
  double buffer[10];

  while (true) {

    udp.receive(buffer, sizeof(buffer));

    // buffer contains input
    inversion.control(buffer);
    // buffer now contains output

    udp.send(buffer, sizeof(buffer));

  }



}
