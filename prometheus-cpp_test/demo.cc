
#include <chrono>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <string>
#include <thread>
#include <time.h>

#include "prometheus/client_metric.h"
#include "prometheus/counter.h"
#include "prometheus/exposer.h"
#include "prometheus/family.h"
#include "prometheus/gateway.h"
#include "prometheus/gauge.h"
#include "prometheus/histogram.h"
#include "prometheus/registry.h"
#include "prometheus/summary.h"

int GetRand(int max) {
  srand((unsigned)time(NULL));
  return rand() % max;
}

int main() {
  using namespace prometheus;

  // create an http server running on bind_addr
  Exposer exposer{"0.0.0.0:8012"};

  // create a metrics registry with component=main labels applied to all its
  // metrics
  auto registry = std::make_shared<Registry>();

  exposer.RegisterCollectable(registry, "/metrics");

  // Couter
  // add a new counter family to the registry 
  auto& counter_family = BuildCounter()
                             .Name("api_called_total")
                             .Help("How many is the api called")
                             .Labels({{"prometheus_test", "sdk_test"}})
                             .Register(*registry);

  // add a counter to the metric family
  auto& api_counter = counter_family.Add(
      {{"prometheus_test_counter", "test_counter"}, {"yet_another_label", "value"}});

  // Gauge
  // add a new gauge family to the registry
  auto& gauge_family = BuildGauge()
                          .Name("cpu_usage")
                          .Help("simulate the cpu usage metric")
                          .Labels({{"prometheus_test", "sdk_test"}})
                          .Register(*registry);

  // add a gauge to the metric family
  auto& cpu_gauge = gauge_family.Add({{"prometheus_test_gauge", "test_gauge"}, {"yet_another_lable", "value"}});


  for (;;) {
    std::this_thread::sleep_for(std::chrono::seconds(10));
    // increment the counter by one (api)
    api_counter.Increment();

    // set the gauge (cpu)
    cpu_gauge.Set(GetRand(100)/100.0);

    // observe the histogram
    // task_histogram.Observe(GetRand(5));

    // observe the summary
    // task_summary.Observe(GetRand(100));

    // push metrics
    // auto returnCode = gateway.Push();
    // std::cout << "returnCode is " << returnCode << std::endl;

    // // statics
    // if (returnCode < 0)
    // {
    //   failTestCount++;
    // }
    // totalTestCount++;
    // std::cout << "total count=" << totalTestCount << ", fail count=" << failTestCount << ", succ_rate=" << (totalTestCount-failTestCount) * 100.f / totalTestCount << "%" << std::endl;
  }

  return 0;
}