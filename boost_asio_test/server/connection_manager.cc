#include "connection_manager.h"

namespace http {
namespace server {

ConnectionManager::ConnectionManager() {}

void ConnectionManager::Start(ConnectionPtr c) {
  connections_.insert(c);
  c->Start();
}

void ConnectionManager::Stop(ConnectionPtr c) {
  connections_.erase(c);
  c->Stop();
}

void ConnectionManager::StopAll() {
  for (auto c : connections_) {
    c->Stop();
  }
  connections_.clear();
}

} // namespace server
} // namespace http