#include "udp.h"

#include <arpa/inet.h>
#include <cstring>
#include <iostream>

/* constructor */
/* local_port: local port, to send from and receive on
  remote_port: remote port, send only, no restrictions on rx */
UDP::UDP(int local_port, int remote_port, std::string remote_ip, int timeout_ms) {

  /* create UDP socket */
  if ( (fd_ = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) {
    std::cout <<  "UDP: couldn't create UDP socket" << std::endl;
    return;
  }

  /* set timout options */
  /*    tv_usec=0 is interpreted as inf, use 1 ns instead */
  struct timeval tv;
  tv.tv_sec = 0;
  tv.tv_usec = timeout_ms==0 ? 1 : timeout_ms*1000;
  if (setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
    std::cout << "UDP: couldn't set socket timeout" << std::endl;
  }

  /* bind socket to any valid IP address and a specific port */
  memset( (void *) &myaddr_, 0, sizeof(myaddr_) ); // initializing to zero
  myaddr_.sin_family = AF_INET;
  myaddr_.sin_addr.s_addr = htonl(INADDR_ANY);
  myaddr_.sin_port = htons(local_port);
  if ( bind(fd_, (struct sockaddr *) &myaddr_, sizeof(myaddr_)) < 0) {
    std::cout << "UDP: local port: " << ntohs(myaddr_.sin_port) << std::endl;
    std::cout << "UDP: couldn't bind to socket" << std::endl;
    return;
  }

  /* setup remote sockaddr */
  memset( (void *) &remaddr_send_, 0, sizeof(remaddr_send_) );
  remaddr_send_.sin_family = AF_INET;
  remaddr_send_.sin_port = htons(remote_port);
  if( inet_aton(remote_ip.data(), &remaddr_send_.sin_addr)==0 ){
    std::cout << "UDP: inet_aton() failed" << std::endl;
  }

  /* print status */
  std::cout << "UDP: initialization complete" << std::endl;
  std::cout << "UDP: sending/receiving on local IP: " << inet_ntoa(myaddr_.sin_addr) << std::endl;
  std::cout << "UDP: sending/receiving on local port: " << ntohs(myaddr_.sin_port) << std::endl;
  std::cout << "UDP: sending to remote IP: " << inet_ntoa(remaddr_send_.sin_addr) << std::endl;
  std::cout << "UDP: sending to remote port: " << ntohs(remaddr_send_.sin_port) << std::endl;
}

int UDP::receive(void* buffer, size_t len) {
  socklen_t remote_len = sizeof(remaddr_recv_);
  int bytes_rec = recvfrom(fd_, buffer, len, 0, (struct sockaddr *) &remaddr_recv_, &remote_len);
  if (verbose_ and bytes_rec != -1) {
    std::cout << "UDP: Received packet of " << bytes_rec << "bytes from IP: "
        << inet_ntoa(remaddr_recv_.sin_addr) << ", port: " << ntohs(myaddr_.sin_port) << std::endl;
  }
  return bytes_rec;
}

void UDP::send(void* buffer, size_t len) {
  socklen_t remote_len = sizeof(remaddr_send_);
  if( sendto(fd_, buffer, len, 0, (struct sockaddr*) &remaddr_send_, remote_len)==-1 ) {
    std::cout << "UDP: sendto failed" << std::endl;
  } else if (verbose_) {
    std::cout << "UDP: sent packet of " << len << "bytes to IP: "
        << inet_ntoa(remaddr_send_.sin_addr) << ", port: " << ntohs(remaddr_send_.sin_port)
        << std::endl;
  }
}

UDP::~UDP() {
  //close(fd_);
  /* Do not close socket! Copies of UDP object will be invalidated. Perhaps the best thing to do
     would be to write a hacky assignment operator to close the socket on the destination object,
     then copy the member variables. For now, just don't close the socket on destruction
   */
}

