#include "HelloWorld.h"
#include "HelloWorldPubSubTypes.h"

#include <chrono>
#include <cstdint>
#include <fastdds/dds/core/status/PublicationMatchedStatus.hpp>
#include <fastdds/dds/domain/qos/DomainParticipantQos.hpp>
#include <fastdds/dds/publisher/qos/DataWriterQos.hpp>
#include <fastdds/dds/publisher/qos/PublisherQos.hpp>
#include <fastdds/dds/topic/qos/TopicQos.hpp>
#include <iostream>
#include <thread>

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/DataWriterListener.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/topic/Topic.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>

using namespace eprosima::fastdds::dds;

class HelloWorldPublisher {

public:
  HelloWorldPublisher()
      : participant_(nullptr), publisher_(nullptr), topic_(nullptr),
        writer_(nullptr), type_(new HelloWorldPubSubType()) {}

  ~HelloWorldPublisher() {
    if (writer_ != nullptr) {
      publisher_->delete_datawriter(writer_);
    }
    if (publisher_ != nullptr) {
      participant_->delete_publisher(publisher_);
    }
    if (topic_ != nullptr) {
      participant_->delete_topic(topic_);
    }
    DomainParticipantFactory::get_instance()->delete_participant(participant_);
  }

  bool Init() {
    hello_.index(0);
    hello_.message("HelloWorld");

    DomainParticipantQos participantQos;
    participantQos.name("Participant_Publisher");
    participant_ = DomainParticipantFactory::get_instance()->create_participant(
        0, participantQos);

    if (participant_ == nullptr) {
      return false;
    }

    type_.register_type(participant_);

    topic_ = participant_->create_topic("HelloWorldTopic", "HelloWorld",
                                        TOPIC_QOS_DEFAULT);
    if (topic_ == nullptr) {
      return false;
    }

    publisher_ = participant_->create_publisher(PUBLISHER_QOS_DEFAULT, nullptr);
    if (publisher_ == nullptr) {
      return false;
    }

    writer_ = publisher_->create_datawriter(topic_, DATAWRITER_QOS_DEFAULT,
                                            &listener_);
    if (writer_ == nullptr) {
      return false;
    }

    return true;
  }

  bool publish() {
    if (listener_.matched_ > 0) {
      hello_.index(hello_.index() + 1);
      writer_->write(&hello_);
      return true;
    }
    return false;
  }

  void run(uint32_t samples) {
    uint32_t samples_sent = 0;
    while (samples_sent < samples) {
      if (publish()) {
        samples_sent++;
        std::cout << "Message: " << hello_.message()
                  << " with index: " << hello_.index() << " SENT" << std::endl;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }

private:
  HelloWorld hello_;

  DomainParticipant* participant_;

  Publisher* publisher_;

  Topic* topic_;

  DataWriter* writer_;

  TypeSupport type_;

  class PubListener : public DataWriterListener {
  public:
    PubListener() : matched_(0) {}
    ~PubListener() override {}

    void on_publication_matched(DataWriter*,
                                const PublicationMatchedStatus& info) override {
      if (info.current_count_change == 1) {
        matched_ = info.total_count;
        std::cout << "Publisher matched" << std::endl;
      } else if (info.current_count_change == -1) {
        matched_ = info.total_count;
        std::cout << "Publisher unmatched" << std::endl;
      } else {
        std::cout << info.current_count_change
                  << " is not a valid value for PublicationMatchedStatus "
                     "current count change."
                  << std::endl;
      }
    }

    std::atomic_int matched_;
  } listener_;
};

int main() {
  std::cout << "Starting publisher." << std::endl;
  uint32_t samples = 10;

  HelloWorldPublisher* mypub = new HelloWorldPublisher();
  if (mypub->Init()) {
    mypub->run(samples);
  }

  delete mypub;
  return 0;
}