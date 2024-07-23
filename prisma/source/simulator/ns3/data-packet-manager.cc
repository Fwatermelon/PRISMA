/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo and Lucile Sassatelli
 *
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Redha A. Alliche, <alliche@i3s.unice.fr,>
 * Author: Tiago Da Silva Barros    <tiago.da-silva-barros@inria.fr>
 * Author: Ramon Aparicio-Pardo       <raparicio@i3s.unice.fr,>
 * Author: Lucile Sassatelli       <sassatelli@i3s.unice.fr,>
 *
 * Université Côte d’Azur, CNRS, I3S, Inria Sophia Antipolis, France
 *
 * Work supported in part by he  support  of  the  French  Agence  Nationale  dela Recherche (ANR), 
 * under grant ANR-19-CE-25-0001-01 (ARTIC project).
 * This  work  was  performed  using  HPC  resources  from  GENCI-IDRIS  (Grant2021-AD011012577).
 *
 */
#include "data-packet-manager.h"
#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "point-to-point-net-device.h"
#include "my-tag.h"
#include "ns3/ppp-header.h"
#include "ns3/csma-net-device.h"
#include "ns3/csma-module.h"

//#include "ns3/wifi-module.h"
#include "ns3/node-list.h"
#include "ns3/log.h"
#include <sstream>
#include <iostream>
#include <vector>
#include <fstream>



namespace ns3 {

template<typename T>
double getAverage(std::vector<T> const& v) {
  if (v.empty()) {
    return 0;
  }

  double sum = 0.0;
  for (const T &i: v) {
    sum += (double)i;
  }
  return sum / v.size();
}

template<typename T>
double getSum(std::vector<T> const& v) {
  if (v.empty()) {
    return 0;
  }
 double sum = 0.0;
  for (const T &i: v) {
    sum += (double)i;
  }
  return sum;
}

NS_LOG_COMPONENT_DEFINE ("DataPacketManager");

DataPacketManager::DataPacketManager ()
{
  PacketManager();
  NS_LOG_FUNCTION (this);
}

void
DataPacketManager::dropPacket(DataPacketManager *entity, Ptr<const Packet> packet){
  MyTag tagCopy;
  packet->PeekPacketTag(tagCopy);
  if ((PacketType(tagCopy.GetSimpleValue()) == DATA_PACKET) && (tagCopy.GetTrafficValable()) && (tagCopy.GetFinalDestination() != entity->m_node->GetId()) && tagCopy.GetLastHop()==entity->m_node->GetId()){
    SentPacket packetLost;
    packetLost.start_time = tagCopy.GetStartTime();
    packetLost.type = DATA_PACKET;
    packetLost.uid = packet->GetUid();
    entity->m_lostPackets.push_back(packetLost);
  }
}

DataPacketManager::DataPacketManager (Ptr<Node> node, vector<int> neighbors, NodeContainer nodes_switch, int controller_id) : PacketManager(node, neighbors)
{
  for(uint32_t j=0;j<nodes_switch.GetN();j++){
    for(uint32_t i=2;i<nodes_switch.Get(j)->GetNDevices();i++){
      nodes_switch.Get(j)->GetDevice(i)->TraceConnectWithoutContext("MacTxDrop", MakeBoundCallback(&dropPacket, this));
    }
  }
  m_nodes_switch = nodes_switch;
  m_controller_address = controller_id;
  NS_LOG_FUNCTION (this);
}

void 
DataPacketManager::setSmallSignalingPacketSize(uint32_t signPacketSize){
  m_signPacketSize = signPacketSize;
}

void 
DataPacketManager::setPingPacketIntervalTime(float pingBackIntervalTime){
  m_pingPacketInterval = Seconds(pingBackIntervalTime);
  // NS_LOG_UNCOND("DataPacketManager::setPingPacketIntervalTime: pingBackIntervalTime="<<pingBackIntervalTime << " at time="<< int(m_obs_bufferLength));
  // if (m_obs_bufferLength){
    // NS_LOG_UNCOND("DataPacketManager::setPingPacketIntervalTime: pingBackIntervalTime="<<pingBackIntervalTime << " at time="<< int(m_obs_bufferLength));
  Simulator::Schedule(m_pingPacketInterval, &DataPacketManager::sendPingPackets, this);
  // }

}

void 
DataPacketManager::setPingBackPacketManager(PingBackPacketManager *pingBackPacketManager){
  m_pingBackPacketManager = pingBackPacketManager;
}

void 
DataPacketManager::setObsBufferLength(bool value){
  m_obs_bufferLength = value;
}


Ptr<OpenGymSpace>
DataPacketManager::getObservationSpace()
{
  NS_LOG_FUNCTION (this);
  uint32_t low = 0;
  uint32_t high = 16260; // max buffer size --> to change depending on actual value (access to defaul sim param)
  m_obs_shape = {uint32_t(1)+uint32_t(m_neighbors.size()),}; // Destination Node + (n_neighbors) interfaces for other nodes
  std::string dtype = TypeNameGet<uint32_t> ();
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, m_obs_shape, dtype);
  return space;
}

vector<uint32_t> 
DataPacketManager::getObsShape(){
  return m_obs_shape;
}


Ptr<OpenGymSpace>
DataPacketManager::getActionSpace()
{
  NS_LOG_FUNCTION (this);
  Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace> (m_neighbors.size()); // first dev is not p2p
  return space;
}

bool 
DataPacketManager::receivePacket(Ptr<Packet> packet, Ptr<NetDevice> receivingNetDev){
  bool ret = PacketManager::receivePacket(packet);
  m_receivingNetDev = receivingNetDev;
  MyTag tagCopy;
  m_packet->PeekPacketTag(tagCopy);
  return tagCopy.GetTrafficValable() && ret;
}

Ptr<OpenGymDataContainer>
DataPacketManager::getObservation()
{ 
  Ptr<OpenGymBoxContainer<uint32_t> > box = CreateObject<OpenGymBoxContainer<uint32_t> >(m_obs_shape);
  //Adding destination to obs
  box->AddValue(m_map_overlay_array[m_destination]);
  

  //Preparing the config
  Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4>();
  ns3::Socket::SocketErrno sockerr;
  Ptr<Ipv4RoutingProtocol> routing = ipv4->GetRoutingProtocol( );
  Ptr<Packet> packet_test = Create<Packet>(20);

  
  //Get Neighbors for acessing their respective queues
  uint32_t value;
  for (size_t i=0 ; i<m_neighbors.size(); i++){
    string string_ip= "10.2.2."+std::to_string(m_neighbors[i]+1);
    Ipv4Address ip_test(string_ip.c_str());
    m_packetIpHeader.SetDestination(ip_test);
    packet_test->AddHeader(m_packetIpHeader);
    Ptr<Ipv4Route> route = routing->RouteOutput (packet_test, m_packetIpHeader, 0, sockerr);
    Ptr<PointToPointNetDevice> dev = DynamicCast<PointToPointNetDevice>(route->GetOutputDevice());
    // NS_LOG_UNCOND("DataPacketManager::getObservation: dev->GetIfIndex()="<<dev->GetIfIndex() << " " << int(m_obs_bufferLength));
    if(m_obs_bufferLength){

      value = this->getQueueLengthInBytes (m_node, dev->GetIfIndex());
      // NS_LOG_UNCOND("dd="<<dev->GetIfIndex() << " " << value << " " << m_node->GetId());
    } else{
      value =uint32_t(1000000*std::max(getAverage(m_pingBackPacketManager->m_tunnelsDelay[i]), double(m_pingBackPacketManager->getMaxTimePingForwardPacketSent(i))));
    }
    box->AddValue(value);
  }

  // Go through all the nodes and get the number of packets in the queue
  // int totalPackets = 0;
  // int totalBytes = 0;
  // // NS_LOG_UNCOND("Node " << m_node->GetId() << " has " << m_neighbors.size() << " neighbors" << " at time " << Simulator::Now().GetSeconds() << "s : " << m_nodes_switch.GetN());
  // for(uint32_t j=0;j<m_nodes_switch.GetN();j++){
  //   for(uint32_t i=2;i<m_nodes_switch.Get(j)->GetNDevices();i++){
  //       Ptr<NetDevice> netDev = m_nodes_switch.Get(j)->GetDevice(i);
  //       Ptr<PointToPointNetDevice> p2p_netDev = DynamicCast<PointToPointNetDevice> (netDev);
  //       Ptr<Queue<Packet> > queue = p2p_netDev->GetQueue ();
  //       totalPackets = totalPackets + (int) queue->GetNPackets();
  //       totalBytes = totalBytes + (int) queue->GetNBytes();
  //       // show the packet id in the queue
  //       for (int k = 0; k < (int) queue->GetNPackets(); k++) {
  //         Ptr<Packet> packet = queue->Dequeue();
  //         // NS_LOG_UNCOND("Node " << m_node->GetId() << " has packet " << packet->GetUid() << " in the queue at time " << Simulator::Now().GetSeconds() << "s" << " with size " << packet->GetSize() << " bytes" );
  //         queue->Enqueue(packet);
  //       }
    
  //   }
  // }
  // NS_LOG_UNCOND("Total Packets in the network: " << totalPackets << " packets and " << totalBytes << " bytes at time " << Simulator::Now().GetSeconds() << "s");

  return box;
}

uint32_t
DataPacketManager::getQueueLengthInBytes(Ptr<Node> node, uint32_t netDev_idx)
{
  Ptr<NetDevice> netDev = node->GetDevice (netDev_idx);
  Ptr<PointToPointNetDevice> p2p_netDev = DynamicCast<PointToPointNetDevice> (netDev);
  Ptr<Queue<Packet> > queue = p2p_netDev->GetQueue ();
  uint32_t backlog = (int) queue->GetNBytes();
  return backlog;
}

float
DataPacketManager::getReward()
{
  return 1;
}

bool
DataPacketManager::getGameOver(){
  return m_destination == m_node->GetId();
}

string
DataPacketManager::getInfo()
{
  string myInfo = PacketManager::getInfo();
  myInfo += ", Packet Lost="; //20
  while (!m_lostPackets.empty())
  {
    SentPacket lostPacket = m_lostPackets.back();
    myInfo += std::to_string(lostPacket.uid) + ";";
    m_lostPackets.pop_back();
  }
  myInfo += ", Source="; //21
  if (m_map_overlay_array.size() > m_source && m_map_overlay_array.size() > m_destination && m_map_overlay_array.size() > m_node->GetId()){
    myInfo += std::to_string(m_map_overlay_array[m_source]);
    myInfo += ", Destination="; //22
    myInfo += std::to_string(m_map_overlay_array[m_destination]);
    myInfo += ", node="; //23
    myInfo += std::to_string(m_map_overlay_array[m_node->GetId()]);
  }
  // NS_LOG_UNCOND(myInfo);
  return myInfo;
}

bool 
DataPacketManager::sendPacket(uint32_t action){
  
  //Get discrete action
  // Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);
  // uint32_t fwdDev_idx = discrete->GetValue();
  uint32_t fwdDev_idx = action;
  if(m_arrivedAtFinalDest){
    // string string_ip = "0.0.0.0";
    // Ipv4Address ip_dest(string_ip.c_str());
    // m_receivingNetDev->Send(m_packet, ip_dest, 0x0800);
    // Do Not Here
    // NS_LOG_UNCOND("PacketRoutingEnv::sendPacket: Packet already arrived at destination" << " at time="<<Simulator::Now().GetSeconds() << " and node " << m_node->GetId() << " and destination " << m_destination << " and source " << m_source << " and last hop " << m_lastHop << " and packet uid " << m_packetUid);
  } else if (fwdDev_idx < m_neighbors.size()){
    // NS_LOG_UNCOND("PacketRoutingEnv::sendPacket: Packet doing hop at node" << " at time="<<Simulator::Now().GetSeconds() << " and node " << m_node->GetId() << " and destination " << m_destination << " and source " << m_source << " and last hop " << m_lastHop << " and packet uid " << m_packetUid);
    //Setting packet Tag
    MyTag sendingTag;
    m_packet->PeekPacketTag(sendingTag);
    sendingTag.SetLastHop(m_node->GetId());
    sendingTag.SetNextHop(m_neighbors[fwdDev_idx]);
    m_packet->ReplacePacketTag(sendingTag);
    
    //Adding Headers
    m_packet->AddHeader(m_packetUdpHeader);
    string string_ip= "10.2.2."+std::to_string(m_neighbors[fwdDev_idx]+1);
    Ipv4Address ip_dest(string_ip.c_str());
    m_packetIpHeader.SetDestination(ip_dest);
    m_packetIpHeader.SetSource(string("10.1.1."+std::to_string(m_node->GetId()+1)).c_str());
    m_packet->AddHeader(m_packetIpHeader);
    
    //Discovering the output buffer based on the routing table
    //(map between overlay tunnel and netDevice)
    Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4>();
    ns3::Socket::SocketErrno sockerr;
    Ptr<Ipv4RoutingProtocol> routing = ipv4->GetRoutingProtocol( );
    Ptr<Ipv4Route> route = routing->RouteOutput (m_packet, m_packetIpHeader, 0, sockerr); 
    route->SetSource(string("10.1.1."+std::to_string(m_node->GetId()+1)).c_str());
    Ptr<PointToPointNetDevice> dev = DynamicCast<PointToPointNetDevice>(route->GetOutputDevice());

    //Send and verify if the Packet was dropped
    m_counterSentPackets += 1;
    SentPacket sent;
    sent.uid = m_packet->GetUid();
    sent.type = DATA_PACKET;
    sent.start_time= Simulator::Now().GetMilliSeconds();
    m_sentPackets.push_back(sent);
    // NS_LOG_UNCOND("DataPacketManager::sendPacket: "<<m_node->GetId() << " " << m_packet->GetUid() << " " << m_packet->GetSize() << " " << m_packet->GetUid() << " " << m_lastHop << " " << m_nextHop << " " << m_destination << " " << m_controller_address << " " << fwdDev_idx);
    dev->Send(m_packet, dev->GetBroadcast(), 0x0800);

  } else{
    //TODO: Implement DropPacket Function
    // dropPacket(this, m_packet);
  }
  return true;
}

void 
DataPacketManager::sendSmallSignalingPacket(bool isFirst){
  // //Don't send if it is the source node
  // if(m_source == m_node->GetId()){
  //   return ;
  // }
  //Define Tag
  MyTag tagSmallSignaling;

  //Define packet size
  double packetSize=m_signPacketSize;
  uint32_t destination = m_lastHop;
  if (m_controller_address!= -1){
    destination = uint32_t(m_controller_address);
    // NS_LOG_UNCOND("PacketRoutingEnv::sendSmallSignalingPacket: packetSize= "<<packetSize<< " at time="<<Simulator::Now().GetSeconds()<< " "<< m_node->GetId()<< " "<< m_lastHop<< " "<< m_packetUid << " " << m_nextHop << " " << m_destination << " " << m_controller_address);
  }

  //Setting Packet Tag
  Ptr<Packet> smallSignalingPckt = Create<Packet> (packetSize);
  tagSmallSignaling.SetSimpleValue(REPLAY_MEMORY_UPDATE_PACKET);
  tagSmallSignaling.SetFinalDestination(destination);
  tagSmallSignaling.SetNextHop(destination);
  tagSmallSignaling.SetLastHop(m_node->GetId());
  tagSmallSignaling.SetSource(m_node->GetId());
  if (isFirst){
    // NS_LOG_UNCOND("DataPacketManager::sendSmallSignalingPacket: isFirst= "<<isFirst<< " at time="<<Simulator::Now().GetSeconds()<< " "<< m_node->GetId()<< " "<< m_lastHop<< " "<< m_packetUid << " " << m_nextHop << " " << m_destination << " " << m_controller_address);
    tagSmallSignaling.SetSourceOverlay(uint32_t(m_map_overlay_array[m_node->GetId()]));
  }
  else {
    // NS_LOG_UNCOND("DataPacketManager::sendSmallSignalingPacket: isFirst= "<<isFirst<< " at time="<<Simulator::Now().GetSeconds()<< " "<< m_node->GetId()<< " "<< m_lastHop<< " "<< m_packetUid << " " << m_nextHop << " " << m_destination << " " << m_controller_address);
   tagSmallSignaling.SetSourceOverlay(uint32_t(m_map_overlay_array[m_lastHop]));
  }
  tagSmallSignaling.SetIsFirst(uint8_t(isFirst));
  tagSmallSignaling.SetIdValue(m_packetUid);

  smallSignalingPckt->AddPacketTag(tagSmallSignaling);
  //Adding headers
  UdpHeader udp_head;
  smallSignalingPckt->AddHeader(udp_head);
  
  Ipv4Header ip_head;
  string string_ip_src= "10.2.2."+std::to_string(m_node->GetId()+1);
  Ipv4Address ip_src(string_ip_src.c_str());
  ip_head.SetSource(ip_src);
  string string_ip_dest;
  string_ip_dest= "10.2.2."+std::to_string(destination+1);
  Ipv4Address ip_dest(string_ip_dest.c_str());
  ip_head.SetDestination(ip_dest);
  ip_head.SetPayloadSize(m_signPacketSize+udp_head.GetSerializedSize());
  ip_head.SetProtocol(17);
  smallSignalingPckt->AddHeader(ip_head);

  // store signalling packet size
  Ptr<PointToPointNetDevice> dev = DynamicCast<PointToPointNetDevice>(m_receivingNetDev);
  dev->m_computeStats->addGlobalBytesSignaling(smallSignalingPckt->GetSize());
  // NS_LOG_UNCOND("DataPacketManager::sendSmallSignalingPacket: "<<m_node->GetId() << " to " << destination << " " << m_packetUid << " " << m_nextHop << " " << m_destination << " " << smallSignalingPckt->GetSize());
  

  //Send the sign packet
  m_receivingNetDev->Send(smallSignalingPckt, m_destAddr, 0x800);
}

void
DataPacketManager::sendPingPackets(){
  m_pingBackPacketManager->addSentPingForwardPacket(uint64_t(m_pingPacketIndex), Simulator::Now().GetMilliSeconds());
  for(uint32_t i = 0; i<m_neighbors.size();i++){
    sendPingForwardPacket(i);
  }
  m_pingPacketIndex += 1;
  Simulator::Schedule(m_pingPacketInterval, &DataPacketManager::sendPingPackets, this);
}

void 
DataPacketManager::sendPingForwardPacket(uint32_t overlayIndex){
  //Define Tag
  MyTag tagPingForward;

  //Define packet size
  double packetSize=8;
  uint32_t destination = m_neighbors[overlayIndex];
  

  
  //Setting Packet Tag
  Ptr<Packet> pingForwardPckt = Create<Packet> (packetSize);
  tagPingForward.SetSimpleValue(uint8_t(PING_FORWARD_PACKET));
  tagPingForward.SetFinalDestination(destination);
  tagPingForward.SetNextHop(destination);
  tagPingForward.SetLastHop(m_node->GetId());
  tagPingForward.SetSource(m_node->GetId());
  tagPingForward.SetStartTime(Simulator::Now().GetMilliSeconds());
  tagPingForward.SetTunnelOverlaySendingIndex(overlayIndex);
  tagPingForward.SetOverlayIndex(m_pingPacketIndex);
 
  //tagSmallSignaling.SetIdValue(m_packetUid);
  //tagSmallSignaling.SetTrafficValable(0);

  pingForwardPckt->AddPacketTag(tagPingForward);

  //Adding headers
  UdpHeader udp_head;
  pingForwardPckt->AddHeader(udp_head);
  
  Ipv4Header ip_head;
  string string_ip_src= "10.2.2."+std::to_string(m_node->GetId()+1);
  Ipv4Address ip_src(string_ip_src.c_str());
  ip_head.SetSource(ip_src);
  string string_ip_dest;
  string_ip_dest= "10.1.1."+std::to_string(destination+1);
  Ipv4Address ip_dest(string_ip_dest.c_str());
  ip_head.SetDestination(ip_dest);
  ip_head.SetPayloadSize(8+udp_head.GetSerializedSize());
  ip_head.SetProtocol(17);
  pingForwardPckt->AddHeader(ip_head);
  
  //Discovering the output buffer based on the routing table
  //(map between overlay tunnel and netDevice)
  Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4>();
  ns3::Socket::SocketErrno sockerr;
  Ptr<Ipv4RoutingProtocol> routing = ipv4->GetRoutingProtocol( );
  Ptr<Ipv4Route> route = routing->RouteOutput (pingForwardPckt, ip_head, 0, sockerr);
  Ptr<PointToPointNetDevice> dev = DynamicCast<PointToPointNetDevice>(route->GetOutputDevice());
  dev->m_computeStats->addGlobalBytesSignaling(pingForwardPckt->GetSize());
  // add it to signalling
  // addGlobalBytesSignaling
  //Send the ping Forward packet
  // NS_LOG_UNCOND("sendPingForwardPacket: "<<m_node->GetId() << " to " << destination << " " << overlayIndex << " " << m_pingPacketIndex << pingForwardPckt->GetSize());

  dev->Send(pingForwardPckt, m_destAddr, 0x800);
}

}// ns3 namespace