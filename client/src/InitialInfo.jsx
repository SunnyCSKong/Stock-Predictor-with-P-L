import React, { useState, useEffect } from 'react';
import fuzzysort from 'fuzzysort';
import './InitialInfo.css'
import APIService from './APIService';
import Results from './Results';

function InitialInfo(props) {
  const [listings, setListings] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [showResults, setShowResults] = useState(true);
  const [average,setAverage] = useState('')
  const [formSubmitted,setFormSubmitted] = useState(false)
  const [responseData, setResponseData] = useState(null); // To store response data
  //get ticker symbols
  useEffect(() => {
    async function fetchListings() {
      try {
        const response = await fetch('/listings');
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.json();
        setListings(data);
      } catch (error) {
        console.error('Error fetching listings:', error);
      }
    }
    fetchListings();
  }, []);


  const handleSearch = (event) => {
    setSearchTerm(event.target.value);
    setShowResults(true);
  };

  const handleListingClick = (symbol) => {
    setSearchTerm(symbol);
    setShowResults(false);
  };

  const filteredListings = fuzzysort
    .go(searchTerm, listings, { key: 'symbol' })
    .splice(0,3)
    .map((result) => result.obj);

    const handleSubmit=(event)=>{
      event.preventDefault();
      setFormSubmitted(true);
      insertSelection();
    };
    const insertSelection = async () => {
      try {
        const response = await APIService.InsertSelections({searchTerm,average});
        setResponseData(response); // Store the response data
      } catch(error) {
        console.log('error:', error);
      }
    };

    const handleSelect=(event)=>{
      setAverage(event.target.value);
    };

  return (
    <div className='searchContainer'>
      <div className="init-container">
      <h1 className='header'>Ticker Search</h1>
        <form onSubmit = {handleSubmit} className='infoForm'>
          <input
            type="text"
            className='searchBar'
            placeholder="Search by ticker symbol/company name"
            value={searchTerm}
            onChange={handleSearch}
          />

          {showResults && (
            <ul className='list'>
              <div className="listings">
              {filteredListings.map((listing, index) => (
                <li key={index} className='symbols' onClick={() => handleListingClick(listing.symbol)}>
                  {listing.symbol}: {listing.name}
                </li> 
              ))}
              </div>
            </ul>
          )}
          <div className="dropdown-container">
            <label htmlFor="averages">SMA/EMA: </label>
            <select name="averages" id="averages" onChange= {handleSelect}>
              <option value="SMA">SMA </option>
              <option value="EMA">EMA </option>
            </select>
          </div>
          <button className='submitButton'>Submit</button>
        </form>
        </div>
        <Results formSubmitted={formSubmitted} responseData={responseData}/> 
    </div>
  );
}

export default InitialInfo;