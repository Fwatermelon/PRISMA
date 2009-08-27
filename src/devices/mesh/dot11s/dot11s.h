/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/* 
 * Copyright (c) 2009 IITP RAS
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
 * Author: Pavel Boyko <boyko@iitp.ru>
 * 
 * This is doxygen module description, don't include 
 */
/**
 * \ingroup mesh 
 * \defgroup dot11s IEEE 802.11s draft
 * 
 * \brief IEEE 802.11s (mesh) draft standard implementation
 * 
 * Current model conforms IEEE 802.11s D3.0 draft version and includes
 * Peer Management Protocol and HWMP (routing) Protocol implementations.
 * 
 * The multi-interface (aka multi radio) mesh points are supported as an 
 * extension of IEEE draft version 3.0. Note that corresponding helper
 * creates single interface station by default.
 * \section Dot11s Overview of IEEE 802.11s
 * Implementation of 802.11s draft standard consists of two main parts:
 * Peer management protocol and HWMP - Hybrid Wireless Mesh Protocol.
 * The task of peer management protocol is to open links, detect beacons,
 * and close peer links due to transmission failures or beacon loss.
 * If peer link between sender and receiver does not exist, the packet will be dropped.
 * \subsubsection IE Implementation of Information elements
 * To make a model of 802.11 stack, we need to implement large quantity of information
 * elements and have an ability to add them in a beacon.
 * Also we have to parse beacon and extract proper information elements from it. So, two
 * classes ns3::WifiInformationElement and ns3::WifiInformationElementVector
 * implement this functionality. The former keeps all information of information element and
 * its elementId and has methods ns3::WifiInformationElement::SerializeInformation and
 * ns3::WifiInformationElement::DeserializeInformation.
 * ns3::WifiInformationElementVector adds all stored information elements to a packet.
 * \subsubsection PMP Peer management protocol
 * Peer management protocol consists of three main parts:
 *      - protocol itself ns3::dot11s::PeerManagementProtocol, which keeps all active peer links
 *      and handles all changes of their states and notifies a routing protocol about link failures.
 *      - its MAC plugin ns3::dot11s::PeerManagementProtocolMac which drops packet, if there is no peer link,
 *      and peek all needed information from management frames and information elements from beacons.
 *      - peer link instance ns3::dot11s::PeerLink which keeps finite state machine for each peer link, keeps
 *      beacon loss counter and counter of failed packets to be transmitted.
 *      
 * Procedure of closing peer link is not described detailed in 802.11s draft standard, so the link may be closed by:
 *      - beacon loss (see an appropriate attribute of ns3::dot11s::PeerLink class)
 *      - transmission failure -- when a predefined number of successive packets have failed to transmit,
 *      the link will be closed.
 *
 * Also Peer management protocol is responsible for beacon collision avoidance, because it keeps
 * beacon timing elements from all neighbours.
 * Note, that PeerManagementProtocol is not attached to MeshPointDevice as a routing protocol,
 * but the structure is similar: the upper layer of protocol ns3::dot11s::PeerManagementProtocol
 * and its plugins ns3::dot11s::PeerManagementProtocolMac.
 * 
 * \subsubsection HWMP Hybrid Wireless Mesh Protocol
 * HWMP is implemented in both modes -- reactive and proactive. Also we have implemented an ability
 * to transmit broadcast data and management frames as unicasts (see appropriate attributes).
 * This feature turns off at a station when the number of neighbours of the station is more than a threshold.
 */
