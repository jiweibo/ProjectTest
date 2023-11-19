#pragma once

#include <set>

#include "connection.h"

namespace http {
namespace server {

/// Manages open connections so that they may be cleanly stopped when the server
/// needs to shutdown.
class ConnectionManager {
public:
  ConnectionManager(const ConnectionManager&) = delete;
  ConnectionManager& operator=(const ConnectionManager&) = delete;

  /// Construct a connection manager.
  ConnectionManager();

  /// Add the specified connection to the manager and start it.
  void Start(ConnectionPtr c);

  /// Stop the specified connection.
  void Stop(ConnectionPtr c);

  /// Stop all connections.
  void StopAll();

private:
  /// The he managed connections.
  std::set<ConnectionPtr> connections_;
};

} // namespace server
} // namespace http