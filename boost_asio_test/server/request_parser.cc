
#include <iostream>

#include "request_parser.h"
#include "header.h"
#include "request.h"

namespace http {
namespace server {

RequestParser::RequestParser() : state_(State::method_start) {}

void RequestParser::Reset() { state_ = State::method_start; }

RequestParser::ResultType RequestParser::Consume(Request& req, char input) {
  switch (state_) {
  case State::method_start:
    if (!is_char(input) || is_ctl(input) || is_tspecial(input)) {
      return ResultType::BAD;
    } else {
      state_ = State::method;
      req.method.push_back(input);
      return ResultType::INDETERMINATE;
    }
  case State::method:
    if (input == ' ') {
      state_ = State::uri;
      return ResultType::INDETERMINATE;
    } else if (!is_char(input) || is_ctl(input) || is_tspecial(input)) {
      return ResultType::BAD;
    } else {
      req.method.push_back(input);
      return ResultType::INDETERMINATE;
    }
  case State::uri:
    if (input == ' ') {
      state_ = State::http_version_h;
      return ResultType::INDETERMINATE;
    } else if (is_ctl(input)) {
      return ResultType::BAD;
    } else {
      req.uri.push_back(input);
      return ResultType::INDETERMINATE;
    }
  case State::http_version_h:
    if (input == 'H') {
      state_ = State::http_version_t_1;
      return ResultType::INDETERMINATE;
    } else {
      return ResultType::BAD;
    }
  case State::http_version_t_1:
    if (input == 'T') {
      state_ = State::http_version_t_2;
      return ResultType::INDETERMINATE;
    } else {
      return ResultType::BAD;
    }
  case State::http_version_t_2:
    if (input == 'T') {
      state_ = State::http_version_p;
      return ResultType::INDETERMINATE;
    } else {
      return ResultType::BAD;
    }
  case State::http_version_p:
    if (input == 'P') {
      state_ = State::http_version_slash;
      return ResultType::INDETERMINATE;
    } else {
      return ResultType::BAD;
    }
  case State::http_version_slash:
    if (input == '/') {
      req.http_version_major = 0;
      req.http_version_minor = 0;
      state_ = State::http_version_major_start;
      return ResultType::INDETERMINATE;
    } else {
      return ResultType::BAD;
    }
  case State::http_version_major_start:
    if (is_digit(input)) {
      req.http_version_major = req.http_version_major * 10 + input - '0';
      state_ = State::http_version_major;
      return ResultType::INDETERMINATE;
    } else {
      return ResultType::BAD;
    }
  case State::http_version_major:
    if (input == '.') {
      state_ = State::http_version_minior_start;
      return ResultType::INDETERMINATE;
    } else if (is_digit(input)) {
      req.http_version_major = req.http_version_major * 10 + input - '0';
      return ResultType::INDETERMINATE;
    } else {
      return ResultType::BAD;
    }
  case State::http_version_minior_start:
    if (is_digit(input)) {
      req.http_version_minor = req.http_version_minor * 10 + input - '0';
      state_ = State::http_version_minior;
      return ResultType::INDETERMINATE;
    } else {
      return ResultType::BAD;
    }
  case State::http_version_minior:
    if (input == '\r') {
      state_ = State::expecting_newline_1;
      return ResultType::INDETERMINATE;
    } else if (is_digit(input)) {
      req.http_version_minor = req.http_version_minor * 10 + input - '0';
      return ResultType::INDETERMINATE;
    } else {
      return ResultType::BAD;
    }
  case State::expecting_newline_1:
    if (input == '\n') {
      state_ = State::header_line_start;
      return ResultType::INDETERMINATE;
    } else {
      return ResultType::BAD;
    }
  case State::header_line_start:
    if (input == '\r') {
      state_ = State::expecting_newline_3;
      return ResultType::INDETERMINATE;
    } else if (!req.headers.empty() && (input == ' ' || input == '\t')) {
      state_ = State::header_lws;
      return ResultType::INDETERMINATE;
    } else if (!is_char(input) || is_ctl(input) || is_tspecial(input)) {
      return ResultType::BAD;
    } else {
      req.headers.push_back(header());
      req.headers.back().name.push_back(input);
      state_ = State::header_name;
      return ResultType::INDETERMINATE;
    }
  case State::header_lws:
    if (input == '\r') {
      state_ = State::expecting_newline_2;
      return ResultType::INDETERMINATE;
    } else if (input == ' ' || input == '\t') {
      return ResultType::INDETERMINATE;
    } else if (is_ctl(input)) {
      return ResultType::BAD;
    } else {
      state_ = State::header_value;
      req.headers.back().value.push_back(input);
      return ResultType::INDETERMINATE;
    }
  case State::header_name:
    if (input == ':') {
      state_ = State::space_before_header_value;
      return ResultType::INDETERMINATE;
    } else if (!is_char(input) || is_ctl(input) || is_tspecial(input)) {
      return ResultType::BAD;
    } else {
      req.headers.back().name.push_back(input);
      return ResultType::INDETERMINATE;
    }
  case State::space_before_header_value:
    if (input == ' ') {
      state_ = State::header_value;
      return ResultType::INDETERMINATE;
    } else {
      return ResultType::BAD;
    }
  case State::header_value:
    if (input == '\r') {
      state_ = State::expecting_newline_2;
      return ResultType::INDETERMINATE;
    } else if (is_ctl(input)) {
      return ResultType::BAD;
    } else {
      req.headers.back().value.push_back(input);
      return ResultType::INDETERMINATE;
    }
  case State::expecting_newline_2:
    if (input == '\n') {
      state_ = State::header_line_start;
      return ResultType::INDETERMINATE;
    } else {
      return ResultType::BAD;
    }
  case State::expecting_newline_3:
    if (input == '\n') {
      return (input == '\n') ? RequestParser::ResultType::GOOD
                             : RequestParser::ResultType::BAD;
    }
  default:
    return ResultType::BAD;
  }
}

bool RequestParser::is_char(int c) { return c >= 0 && c <= 127; }

bool RequestParser::is_ctl(int c) { return (c >= 0 && c <= 31) || (c == 127); }

bool RequestParser::is_tspecial(int c) {
  switch (c) {
  case '(':
  case ')':
  case '<':
  case '>':
  case '@':
  case ',':
  case ';':
  case ':':
  case '\\':
  case '"':
  case '/':
  case '[':
  case ']':
  case '?':
  case '=':
  case '{':
  case '}':
  case ' ':
  case '\t':
    return true;
  default:
    return false;
  }
}

bool RequestParser::is_digit(int c) { return c >= '0' && c <= '9'; }

} // namespace server
} // namespace http