import React, {Component, useState, useEffect} from 'react';
import './App.css';
import Table from './components/Table.js';
import ReactPoker from './components/ReactPoker.js';
import { NavigationBar } from './components/Navbar.js';

function App() {
  return (
    <div className="App"> 
      <React.Fragment>

          {/* <NavigationBar />

          <Table /> */}

          <ReactPoker />
      </React.Fragment>
    </div>
  )
}

// class App extends Component {
//   constructor(props) {
//     super(props);
//     this.state = {
//       activeItem: {
//         id: "",
//         name: "",
//         stack: "",
//       },
//       playerList: []
//       };
//   }

//     async componentDidMount() {
//       try {
//         const res = await fetch('http://localhost:8000/api/players/', {
//           headers: {
//             'Content-Type': 'application/json',
//             'Authorization' : 'Token caf80ff750f4fce1d5f58b6f79a90cc0ef47614c'
//           }
//         });
//         const playerList = await res.json();
//         this.setState({
//           playerList 
//         });
//         console.log(playerList)
        
//       } catch (e) {
//         console.log(e);
//     }
//     }
//     renderItems = () => {
      
//       return this.state.playerList.map(item => (
//         <li 
//           key={item.id}
//           className="list-group-item d-flex justify-content-between align-items-center"
//         >
//           <span 
//             className={`todo-title mr-2 ${
//               this.state.viewCompleted ? "completed-todo" : ""
//             }`}
//             title={item.id}
//             >
//               {item.stack}
//             </span>
//         </li>
//       ));
//     };

//     render() {
//       return (
//         <main className="content">
//         <div className="row">
//           <div className="col-md-6 col-sm-10 mx-auto p-0">
//             <div className="card p-3">
//               <ul className="list-group list-group-flush">
//                 {this.renderItems()}
//               </ul>
//             </div>
//           </div>
//         </div>
//       </main>
//       )
//     }
//   }
  
export default App;