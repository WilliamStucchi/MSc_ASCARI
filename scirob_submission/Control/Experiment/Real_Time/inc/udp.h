#pragma once

#include <netdb.h>
#include <string>
#include <sys/socket.h>

class UDP {
  private:
    struct sockaddr_in myaddr_;        // local address
    struct sockaddr_in remaddr_send_;  // remote address to send packets to
    struct sockaddr_in remaddr_recv_;  // remote address, info about where received packets came from
    int fd_;

  public:
    bool verbose_ = false;             // print when sending and receiving

    UDP() {}                           // default constructor, results in invalid udp object
    UDP(int local_port, int remote_port): UDP(local_port, remote_port, "***.****.*.***", 100) {}
    UDP(int local_port, int remote_port, std::string remote_ip, int timeout_ms);
    ~UDP();
    int receive(void* buffer, size_t len);
    void send(void* buffer, size_t len);
};
