#ifndef BAG_TIME_READER__BAG_TIME_READER_HPP_
#define BAG_TIME_READER__BAG_TIME_READER_HPP_

#include <string>
#include <memory>


class BagTimeReader
{
public:
  BagTimeReader(const std::string & bag_path);
  void readMessages();

private:
  std::string bag_path_;
};

#endif  // BAG_TIME_READER__BAG_TIME_READER_HPP_
